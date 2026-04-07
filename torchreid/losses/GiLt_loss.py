# -*- coding: utf-8 -*-
from __future__ import division, absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torchreid.losses import init_part_based_triplet_loss, CrossEntropyLoss
from torchreid.utils.constants import GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS

try:
    from torchmetrics import Accuracy  # noqa: F401
    _HAS_TORCHMETRICS = True
except Exception:
    _HAS_TORCHMETRICS = False


class GiLtLoss(nn.Module):
    """
    Global-identity Local-triplet ('GiLt') 组合损失。
    WACV'23 BPBreID 论文
    """

    default_losses_weights = {
        GLOBAL:       {'id': 1., 'tr': 0.},
        FOREGROUND:   {'id': 1., 'tr': 0.},
        CONCAT_PARTS: {'id': 1., 'tr': 0.},
        PARTS:        {'id': 0., 'tr': 1.},
    }

    def __init__(
        self,
        losses_weights=None,
        use_visibility_scores=False,
        triplet_margin=0.3,
        loss_name='part_averaged_triplet_loss',
        use_gpu=False,
        num_classes=-1,
        writer=None
    ):
        super().__init__()
        if losses_weights is None:
            losses_weights = self.default_losses_weights

        self.use_gpu = use_gpu
        self.losses_weights = losses_weights
        self.use_visibility_scores = use_visibility_scores
        self.writer = writer

        # Triplet（按部件）
        self.part_triplet_loss = init_part_based_triplet_loss(
            loss_name, margin=triplet_margin, writer=writer
        )

        # 带 label smoothing 的 CE
        self.identity_loss = CrossEntropyLoss(label_smooth=True)

        self._warned_label_oob = False

    # ------------------------------- #
    #             前向汇总            #
    # ------------------------------- #
    def forward(self, embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids, only_global=False):
        """
        only_global=True 时：只计算 GLOBAL 的 CE/Triplet
        """
        loss_summary = {}
        losses = []

        # 1) 分类（CE）
        for key in (GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS):
            # 🚀 如果只用全局，则跳过其他分支
            if only_global and key != GLOBAL:
                continue

            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            ce_w = float(self.losses_weights.get(key, {}).get('id', 0.0))

            id_scores = id_cls_scores_dict.get(key, None) if isinstance(id_cls_scores_dict, dict) else None
            vis_scores = visibility_scores_dict.get(key, None) if isinstance(visibility_scores_dict, dict) else None

            if ce_w > 0.0 and id_scores is not None:
                ce_loss, ce_acc = self.compute_id_cls_loss(id_scores, pids, vis_scores)
                losses.append((ce_w, ce_loss))
                loss_info['c'] = ce_loss
                loss_info['a'] = ce_acc

            loss_summary[key] = loss_info

        # 2) Triplet（按部件/或全局）
        for key in (GLOBAL, FOREGROUND, CONCAT_PARTS, PARTS):
            if only_global and key != GLOBAL:
                continue

            loss_info = OrderedDict() if key not in loss_summary else loss_summary[key]
            tr_w = float(self.losses_weights.get(key, {}).get('tr', 0.0))

            emb = embeddings_dict.get(key, None) if isinstance(embeddings_dict, dict) else None
            vis = visibility_scores_dict.get(key, None) if isinstance(visibility_scores_dict, dict) else None

            if tr_w > 0.0 and emb is not None:
                tr_loss, trivial_ratio, valid_ratio = self.compute_triplet_loss(emb, vis, pids)
                losses.append((tr_w, tr_loss))
                loss_info['t'] = tr_loss
                loss_info['tt'] = trivial_ratio
                loss_info['vt'] = valid_ratio

            loss_summary[key] = loss_info

        # 3) 加权求和
        if not losses:
            device = pids.device if isinstance(pids, torch.Tensor) and pids.is_cuda else None
            return torch.tensor(0.0, device=device), loss_summary

        loss = torch.stack([w * l for (w, l) in losses]).sum()
        return loss, loss_summary

    # ------------------------------- #
    #           Triplet 计算           #
    # ------------------------------- #
    def compute_triplet_loss(self, embeddings, visibility_scores, pids):
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        elif embeddings.dim() != 3:
            B = embeddings.size(0)
            embeddings = embeddings.view(B, -1).unsqueeze(1)

        parts_visibility = None
        if self.use_visibility_scores and visibility_scores is not None:
            vis = torch.as_tensor(visibility_scores, device=embeddings.device)
            if vis.dim() == 1:
                vis = vis.unsqueeze(1)
            elif vis.dim() > 2:
                vis = vis.view(vis.size(0), -1)
            P = embeddings.size(1)
            if vis.size(1) != P:
                if vis.size(1) < P:
                    rep = (P + vis.size(1) - 1) // vis.size(1)
                    vis = vis.repeat(1, rep)[:, :P]
                else:
                    vis = vis[:, :P]
            parts_visibility = vis

        try:
            out = self.part_triplet_loss(embeddings, pids, parts_visibility=parts_visibility)
        except TypeError:
            out = self.part_triplet_loss(embeddings, pids)

        if out is None:
            zero = torch.zeros(1, device=embeddings.device, dtype=embeddings.dtype)
            return zero.squeeze(0), 1.0, 0.0

        if isinstance(out, (tuple, list)) and len(out) == 3:
            triplet_loss, trivial_ratio, valid_triplets_ratio = out
        else:
            triplet_loss, trivial_ratio, valid_triplets_ratio = out, 0.0, 1.0

        return triplet_loss, trivial_ratio, valid_triplets_ratio

    # ------------------------------- #
    #           CE（分类）计算          #
    # ------------------------------- #
    def compute_id_cls_loss(self, id_cls_scores, pids, visibility_scores=None):
        if not isinstance(pids, torch.Tensor):
            pids = torch.as_tensor(pids, device=id_cls_scores.device, dtype=torch.long)
        else:
            pids = pids.to(id_cls_scores.device).long()

        if id_cls_scores.dim() == 3:
            B, P, C = id_cls_scores.shape
            logits = id_cls_scores.reshape(B * P, C)
            labels = pids.view(-1, 1).repeat(1, P).reshape(-1)
            vis = None
            if visibility_scores is not None:
                vis = torch.as_tensor(visibility_scores, device=id_cls_scores.device)
                if vis.dtype == torch.bool:
                    vis = vis.float()
                if vis.dim() == 1:
                    vis = vis.unsqueeze(1)
                if vis.dim() == 2 and vis.size(1) == 1:
                    vis = vis.repeat(1, P)
                vis = vis.reshape(B * P)
        elif id_cls_scores.dim() == 2:
            B, C = id_cls_scores.shape
            logits = id_cls_scores
            labels = pids.view(-1)
            vis = None
        else:
            raise ValueError(f"Unsupported id_cls_scores.dim() = {id_cls_scores.dim()}")

        valid = (labels >= 0) & (labels < logits.size(1))
        logits_v = logits[valid]
        labels_v = labels[valid]

        ce_vec = F.cross_entropy(logits_v, labels_v, reduction='none')
        loss = ce_vec.mean()
        acc = float((logits_v.argmax(dim=1) == labels_v).float().mean().item())

        return loss, acc
