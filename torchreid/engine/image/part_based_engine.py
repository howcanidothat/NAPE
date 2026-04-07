from __future__ import division, print_function, absolute_import

import os.path as osp
import torch
import numpy as np
from tabulate import tabulate
from torch.cuda import amp
from tqdm import tqdm

from ..engine import Engine
from ... import metrics
from ...losses.GiLt_loss import GiLtLoss
from ...losses.body_part_attention_loss import BodyPartAttentionLoss
from ...metrics.distance import compute_distance_matrix_using_bp_features
from ...utils import (
    plot_body_parts_pairs_distance_distribution,
    plot_pairs_distance_distribution,
    re_ranking,
)
from torchreid.utils.constants import *

from ...utils.tools import extract_test_embeddings
from ...utils.torchtools import collate
from ...utils.visualization.feature_map_visualization import display_feature_maps


class ImagePartBasedEngine(Engine):
    r"""Training/testing engine for part-based image-reid.
    """

    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        writer,
        loss_name,
        config,
        dist_combine_strat,
        batch_size_pairwise_dist_matrix,
        engine_state,
        margin=0.3,
        scheduler=None,
        use_gpu=True,
        save_model_flag=False,
        mask_filtering_training=False,
        mask_filtering_testing=False,
    ):
        super(ImagePartBasedEngine, self).__init__(
            config,
            datamanager,
            writer,
            engine_state,
            use_gpu=use_gpu,
            save_model_flag=save_model_flag,
            detailed_ranking=config.test.detailed_ranking,
        )

        self.model = model
        self.register_model("model", model, optimizer, scheduler)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parts_num = self.config.model.kpr.masks.parts_num
        self.mask_filtering_training = mask_filtering_training
        self.mask_filtering_testing = mask_filtering_testing
        self.dist_combine_strat = dist_combine_strat
        self.batch_size_pairwise_dist_matrix = batch_size_pairwise_dist_matrix
        self.losses_weights = self.config.loss.part_based.weights
        self.mixed_precision = self.config.train.mixed_precision

        # 修复PyTorch版本兼容性问题
        if self.mixed_precision:
            try:
                # 尝试新版本的路径
                self.scaler = torch.cuda.amp.GradScaler()
            except AttributeError:
                try:
                    # 尝试旧版本的路径
                    self.scaler = torch.amp.GradScaler('cuda')
                except AttributeError:
                    # 如果都不行，禁用混合精度
                    print("Warning: GradScaler not available, disabling mixed precision")
                    self.mixed_precision = False
                    self.scaler = None
        else:
            self.scaler = None

        # Losses
        # 确保num_classes至少为2，满足torchmetrics的要求
        num_classes = max(datamanager.num_train_pids, 200)  # 使用200作为默认值，因为Occluded_REID有200个人
        self.GiLt = GiLtLoss(
            self.losses_weights,
            use_visibility_scores=self.mask_filtering_training,
            triplet_margin=margin,
            loss_name=loss_name,
            writer=self.writer,
            use_gpu=self.use_gpu,
            num_classes=num_classes,
        )

        self.body_part_attention_loss = BodyPartAttentionLoss(
            loss_type=self.config.loss.part_based.ppl,
            use_gpu=self.use_gpu,
            best_pred_ratio=self.config.loss.part_based.best_pred_ratio,
            num_classes=self.parts_num+1,
        )

        # Timers
        self.feature_extraction_timer = self.writer.feature_extraction_timer
        self.loss_timer = self.writer.loss_timer
        self.optimizer_timer = self.writer.optimizer_timer

        # Camera-related config diagnostics
        try:
            if hasattr(self.config, 'model') and hasattr(self.config.model, 'transreid'):
                trcfg = self.config.model.transreid
                cam_num = getattr(trcfg, 'cam_num', None)
                sie_camera = getattr(trcfg, 'sie_camera', None)
                print(f"[CameraCfg] transreid.cam_num={cam_num}, sie_camera={sie_camera}")
                if sie_camera is False:
                    print("[CameraCfg] sie_camera=False -> 模型通常不会启用相机嵌入，cam_num 对推理无影响（仅作为配置记录）。")
        except Exception as e:
            print(f"[CameraCfg] 读取 transreid 配置失败: {e}")

    def forward_backward(self, data):
        # ===== 取数据 =====
        imgs, target_masks, prompt_masks, keypoints_xyc, pids, imgs_path, cam_id, kp_quality, has_json = self.parse_data_for_train(
            data)

        if self.use_gpu:
            imgs = imgs.cuda()
            cam_id = cam_id.cuda()
            if target_masks is not None: target_masks = target_masks.cuda()
            if prompt_masks is not None: prompt_masks = prompt_masks.cuda()
            if keypoints_xyc is not None: keypoints_xyc = keypoints_xyc.cuda()
            pids = pids.cuda()
            if isinstance(kp_quality, torch.Tensor): kp_quality = kp_quality.cuda()

        # ===== 判定 JSON 是否有效 =====
        valid_json = False
        if keypoints_xyc is not None and keypoints_xyc.numel() > 0:
            if keypoints_xyc.abs().sum().item() > 1e-6:
                valid_json = True

        # ===== 前向 =====
        self.feature_extraction_timer.start()
        outputs = self.model(
            imgs,
            target_masks=target_masks if valid_json else None,
            prompt_masks=prompt_masks if valid_json else None,
            keypoints_xyc=keypoints_xyc if valid_json else None,
            cam_label=cam_id,
        )
        self.feature_extraction_timer.stop()

        # ---- 解包 ----
        def _unpack_train_outputs(out):
            n = len(out)
            if n == 7:
                return out
            elif n == 6:
                e, v, pm, px, sf, idc = out;
                return e, v, pm, px, sf, None, idc
            elif n == 5:
                e, v, pm, px, idc = out;
                return e, v, pm, px, None, None, idc
            elif n == 4:
                e, v, pm, idc = out;
                return e, v, pm, None, None, None, idc
            else:
                raise ValueError(f"Unexpected number of outputs: {n}")

        embeddings_dict, visibility_scores_dict, parts_masks, pixels_cls_scores, spatial_features, masks, id_cls_scores_dict = _unpack_train_outputs(
            outputs)

        # ===== Loss =====
        self.loss_timer.start()
        if valid_json:
            loss, loss_summary = self.combine_losses(
                visibility_scores_dict, embeddings_dict, id_cls_scores_dict, pids,
                pixels_cls_scores, target_masks,
                bpa_weight=self.losses_weights[PIXELS]["ce"]
            )
        else:
            loss, loss_summary = self.GiLt(
                embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids,
                only_global=True  # GiLt 里要实现这个参数
            )
        self.loss_timer.stop()

        # ===== 反传优化 =====
        self.optimizer_timer.start()
        self.optimizer.zero_grad()
        if self.scaler is None:
            loss.backward()
            self.optimizer.step()
        else:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        self.optimizer_timer.stop()

        return loss, loss_summary

    def combine_losses(
        self,
        visibility_scores_dict,
        embeddings_dict,
        id_cls_scores_dict,
        pids,
        pixels_cls_scores=None,
        target_masks=None,
        bpa_weight=0,
    ):
        # 1. ReID objective:
        # GiLt loss on holistic and part-based embeddings
        loss, loss_summary = self.GiLt(
            embeddings_dict, visibility_scores_dict, id_cls_scores_dict, pids
        )

        # 2. Part prediction objective:
        # Body part attention loss on spatial feature map
        if (
            pixels_cls_scores is not None
            and target_masks is not None
            and bpa_weight > 0
        ):
            # resize external masks to fit feature map size
            # target_masks = nn.functional.interpolate(  # FIXME should be useless
            #     target_masks,
            #     pixels_cls_scores.shape[2::],
            #     mode="bilinear",
            #     align_corners=True,
            # )
            # compute target part index for each spatial location, i.e. each spatial location (pixel) value indicate
            # the (body) part that spatial location belong to, or 0 for background.
            pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]  # TODO check first indices are prioretized if equality
            # compute the classification loss for each pixel
            bpa_loss, bpa_loss_summary = self.body_part_attention_loss(
                pixels_cls_scores, pixels_cls_score_targets
            )
            loss += bpa_weight * bpa_loss
            loss_summary = {**loss_summary, **bpa_loss_summary}

        return loss, loss_summary

    def _feature_extraction(self, data_loader):
        f_, pids_, camids_, parts_visibility_, p_masks_, pxl_scores_, anns = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for batch_idx, data in enumerate(tqdm(data_loader, desc=f"Batches processed")):
            imgs, target_masks, prompt_masks, keypoints_xyc, pids, camids = self.parse_data_for_eval(data)
            if self.use_gpu:
                if target_masks is not None:
                    target_masks = target_masks.cuda()
                if prompt_masks is not None:
                    prompt_masks = prompt_masks.cuda()
                imgs = imgs.cuda()
            self.writer.test_batch_timer.start()
            model_output = self.model(imgs, target_masks=target_masks, prompt_masks=prompt_masks, keypoints_xyc=keypoints_xyc, cam_label=camids)
            (
                features,
                visibility_scores,
                parts_masks,
                pixels_cls_scores,
            ) = extract_test_embeddings(
                model_output, self.config.model.kpr.test_embeddings
            )
            self.writer.test_batch_timer.stop()
            if self.mask_filtering_testing:
                parts_visibility = visibility_scores
                parts_visibility = parts_visibility.cpu()
                parts_visibility_.append(parts_visibility)
            else:
                parts_visibility_ = None
            features = features.data.cpu()
            parts_masks = parts_masks.data.cpu()
            f_.append(features)
            p_masks_.append(parts_masks)
            pxl_scores_.append(pixels_cls_scores.data.cpu() if pixels_cls_scores is not None else None)
            pids_.extend(pids)
            camids_.extend(camids)
            anns.append(data)
        if self.mask_filtering_testing:
            parts_visibility_ = torch.cat(parts_visibility_, 0)
        f_ = torch.cat(f_, 0)
        p_masks_ = torch.cat(p_masks_, 0)
        pxl_scores_ = torch.cat(pxl_scores_, 0) if pxl_scores_[0] is not None else None
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        anns = collate(anns)
        return f_, pids_, camids_, parts_visibility_, p_masks_, pxl_scores_, anns

    @torch.no_grad()
    def _evaluate(
            self,
            epoch,
            dataset_name="",
            query_loader=None,
            gallery_loader=None,
            dist_metric="euclidean",
            normalize_feature=False,
            visrank=False,
            visrank_topk=10,
            visrank_q_idx_list=[],
            visrank_count=10,
            save_dir="",
            use_metric_cuhk03=False,
            ranks=[1, 5, 10, 20],
            rerank=False,
            save_features=False,
    ):
        print("Extracting features from query set ...")
        q_res = self._feature_extraction(query_loader)
        if isinstance(q_res, (list, tuple)) and len(q_res) == 7:
            (
                qf,
                q_pids,
                q_camids,
                qf_parts_visibility,
                q_parts_masks,
                q_pxl_scores_,
                q_anns,
            ) = q_res
        elif isinstance(q_res, (list, tuple)) and len(q_res) == 4:
            qf, q_pids, q_camids, q_anns = q_res
            qf_parts_visibility = None
            q_parts_masks = None
            q_pxl_scores_ = None
        else:
            raise ValueError(
                f"Unexpected _feature_extraction() return length: {len(q_res) if isinstance(q_res, (list, tuple)) else type(q_res)}")
        print("Done, obtained {} tensor".format(qf.shape))

        print("Extracting features from gallery set ...")
        g_res = self._feature_extraction(gallery_loader)
        if isinstance(g_res, (list, tuple)) and len(g_res) == 7:
            (
                gf,
                g_pids,
                g_camids,
                gf_parts_visibility,
                g_parts_masks,
                g_pxl_scores_,
                g_anns,
            ) = g_res
        elif isinstance(g_res, (list, tuple)) and len(g_res) == 4:
            gf, g_pids, g_camids, g_anns = g_res
            gf_parts_visibility = None
            g_parts_masks = None
            g_pxl_scores_ = None
        else:
            raise ValueError(
                f"Unexpected _feature_extraction() return length: {len(g_res) if isinstance(g_res, (list, tuple)) else type(g_res)}")
        print("Done, obtained {} tensor".format(gf.shape))

        print(
            "Test batch feature extraction speed: {:.4f} sec/batch".format(
                self.writer.test_batch_timer.avg
            )
        )

        # Eval-time camera distribution and cross-camera positive availability diagnostics
        try:
            q_cam_set = sorted(set(q_camids.tolist())) if hasattr(q_camids, 'tolist') else sorted(set(q_camids))
            g_cam_set = sorted(set(g_camids.tolist())) if hasattr(g_camids, 'tolist') else sorted(set(g_camids))
            print(f"[EvalDiag] Unique cams -> query: {q_cam_set} | gallery: {g_cam_set}")

            from collections import defaultdict as _dd
            g_pid_to_cams = _dd(set)
            for pid, cam in zip(g_pids, g_camids):
                g_pid_to_cams[pid].add(cam)

            cross_cam_ok = 0
            same_cam_only = 0
            no_pos = 0
            for pid, q_cam in zip(q_pids, q_camids):
                cams = g_pid_to_cams.get(pid, None)
                if not cams:
                    no_pos += 1
                elif any(c != q_cam for c in cams):
                    cross_cam_ok += 1
                else:
                    same_cam_only += 1
            print(
                f"[EvalDiag] cross-cam OK: {cross_cam_ok} | same-cam-only (filtered by Market1501 rule): {same_cam_only} | no-pos-in-gallery: {no_pos} | num_q={len(q_pids)}")
        except Exception as e:
            print(f"[EvalDiag] 统计跨相机正样本时出错: {e}")

        # Derive eval_metric robustly
        eval_metric = None
        # 1) prefer config override if available
        eval_metric_cfg = None
        try:
            eval_metric_cfg = getattr(self.config.test, "eval_metric", None)
        except Exception:
            eval_metric_cfg = None
        eval_metric = eval_metric_cfg

        # 2) fallback to dataset-provided metric
        if not eval_metric:
            if hasattr(query_loader, "dataset"):
                eval_metric = getattr(query_loader.dataset, "eval_metric", None)
        if not eval_metric:
            try:
                eval_metric = self.datamanager.test_loader[dataset_name]["query"].dataset.eval_metric
            except Exception:
                eval_metric = "default"

        # 3) alias map (support user-friendly names)
        alias_map = {
            "market1501": "default",
            "Market1501": "default",
            "market": "default",
            "Market": "default",
            # do not filter out same-camera gallery: use generic no-filter metric
            "no_cam_filter": "mot_inter_intra_video",
            "nocam": "mot_inter_intra_video",
            "all": "mot_inter_intra_video",
        }
        eval_metric = alias_map.get(eval_metric, eval_metric)
        print(f"[EvalDiag] Using eval_metric='{eval_metric}' (config.test.eval_metric={eval_metric_cfg})")

        # Optionally save features
        if save_features:
            features_dir = osp.join(save_dir, "features")
            print("Saving features to : " + features_dir)
            torch.save(gf, osp.join(features_dir, f"gallery_features_{dataset_name}.pt"))
            torch.save(qf, osp.join(features_dir, f"query_features_{dataset_name}.pt"))

        self.writer.performance_evaluation_timer.start()
        if normalize_feature:
            print("Normalizing features with L2 norm ...")
            qf = self.normalize(qf)
            gf = self.normalize(gf)

        # Optional visibility binarization for testing and diagnostics
        try:
            cfg_kpr = self.config.model.kpr
            if cfg_kpr.mask_filtering_testing and (qf_parts_visibility is not None) and (
                    gf_parts_visibility is not None):
                if getattr(cfg_kpr, "testing_binary_visibility_score", False):
                    thr = float(getattr(cfg_kpr, "visibility_binary_threshold", 0.5))
                    if qf_parts_visibility.dtype is not torch.bool:
                        qf_parts_visibility = (qf_parts_visibility >= thr)
                    if gf_parts_visibility.dtype is not torch.bool:
                        gf_parts_visibility = (gf_parts_visibility >= thr)

                    def _vis_ratio_bool(t):
                        return float(t.float().mean().item())

                    print(
                        f"[VisBin] testing_binary_visibility_score=True, thr={thr:.2f}, q/g visible_ratio={_vis_ratio_bool(qf_parts_visibility):.3f}/{_vis_ratio_bool(gf_parts_visibility):.3f}")
                else:
                    def _vis_avg(t):
                        return float(t.float().mean().item())

                    print(
                        f"[VisBin] testing_binary_visibility_score=False, q/g avg_vis={_vis_avg(qf_parts_visibility):.3f}/{_vis_avg(gf_parts_visibility):.3f}")
        except Exception as e:
            print(f"[VisBin][WARN] diagnostics failed: {e}")

        # ---------- 保存“原始可见性”在 fallback 之前（关键！）----------
        orig_q_vis = qf_parts_visibility.clone() if qf_parts_visibility is not None else None
        orig_g_vis = gf_parts_visibility.clone() if gf_parts_visibility is not None else None

        # Fallback for samples with no visible parts: use all parts instead of none
        def _apply_empty_mask_fallback(vis, tag):
            try:
                if vis is None:
                    return vis
                empty_mask = (vis.sum(dim=1) == 0)
                num_empty = int(empty_mask.sum().item())
                if num_empty > 0:
                    fill_value = True if vis.dtype is torch.bool else 1.0
                    vis[empty_mask, :] = fill_value
                    print(
                        f"[VisBin][Fallback] {tag}: {num_empty} samples had no visible parts; using all-visible mask for them.")
                return vis
            except Exception as e:
                print(f"[VisBin][Fallback][WARN] Failed to apply fallback on {tag}: {e}")
                return vis

        if getattr(self.config.model.kpr, "empty_visibility_fallback", True):
            qf_parts_visibility = _apply_empty_mask_fallback(qf_parts_visibility, 'query')
            gf_parts_visibility = _apply_empty_mask_fallback(gf_parts_visibility, 'gallery')

        print("Computing distance matrix with metric={} ...".format(dist_metric))

        # ====== PromptGate 融合：仅在 q/g 都有“真实且质量达标”的JSON时叠加部件距离 ======
        from ...metrics.distance import compute_distance_matrix  # 纯全局距离

        def _has_real_vis(vis, eps=1e-6):
            if vis is None: return None
            if vis.dtype is torch.bool:
                return ~(vis.all(dim=1))
            else:
                return ~((vis - 1.0).abs().max(dim=1).values < eps)

        def _quality(vis):
            if vis is None: return None
            if vis.dtype is torch.bool:
                return vis.float().mean(dim=1)
            else:
                return vis.clamp(0, 1).mean(dim=1)

        tau = float(getattr(self.config.model.kpr, "kp_quality_threshold", 0.20))
        alpha = float(getattr(self.config.test, "prompt_fuse_alpha", 0.80))

        q_has = _has_real_vis(orig_q_vis)
        g_has = _has_real_vis(orig_g_vis)
        q_q = _quality(orig_q_vis)
        g_q = _quality(orig_g_vis)
        q_ok = (q_has & (q_q >= tau)) if (q_has is not None and q_q is not None) else None
        g_ok = (g_has & (g_q >= tau)) if (g_has is not None and g_q is not None) else None

        # 纯全局距离
        qf_global = qf.mean(dim=1) if qf.dim() == 3 else qf
        gf_global = gf.mean(dim=1) if gf.dim() == 3 else gf
        dist_global = compute_distance_matrix(qf_global, gf_global, metric=dist_metric)

        # 部件距离
        dist_bp, body_parts_distmat = compute_distance_matrix_using_bp_features(
            qf, gf, qf_parts_visibility, gf_parts_visibility,
            self.dist_combine_strat, self.batch_size_pairwise_dist_matrix,
            self.use_gpu, dist_metric,
        )

        # 融合 or 退回
        dist = dist_global.clone()
        if (q_ok is not None) and (g_ok is not None):
            pair_mask = torch.outer(q_ok.float(), g_ok.float()) > 0.5
            if pair_mask.any():
                dist[pair_mask] = alpha * dist_global[pair_mask] + (1 - alpha) * dist_bp[pair_mask]
                used_ratio = pair_mask.float().mean().item() * 100.0
                print(f"[PromptGate] used_pairs={used_ratio:.2f}% (tau={tau}, alpha={alpha})")
        else:
            print("[PromptGate] no visibility info; using pure global distance.")

        # numpy 化
        distmat = dist.cpu().numpy()
        body_parts_distmat = body_parts_distmat.cpu().numpy()

        # 仅保留一处 re-ranking（用纯全局 qq/gg 更稳）
        if rerank:
            print("Applying person re-ranking ...")
            distmat_qq = compute_distance_matrix(qf_global, qf_global, metric=dist_metric).cpu().numpy()
            distmat_gg = compute_distance_matrix(gf_global, gf_global, metric=dist_metric).cpu().numpy()
            distmat = re_ranking(distmat, distmat_qq, distmat_gg)

        print("Computing CMC and mAP for eval metric '{}' ...".format(eval_metric))
        eval_metrics = metrics.evaluate_rank(
            distmat,
            q_pids,
            g_pids,
            q_camids,
            g_camids,
            q_anns=q_anns,
            g_anns=g_anns,
            eval_metric=eval_metric,
            max_rank=np.array(ranks).max(),
            use_cython=False,
        )

        mAP = eval_metrics["mAP"]
        cmc = eval_metrics["cmc"]
        print("** Results **")
        print("mAP: {:.2%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.2%}".format(r, cmc[r - 1]))

        for metric in eval_metrics.keys():
            if metric not in {"mAP", "cmc", "all_AP", "all_cmc"}:
                print("{:<20}: {}".format(metric, eval_metrics[metric]))

        # Parts ranking
        if self.detailed_ranking:
            self.display_individual_parts_ranking_performances(
                body_parts_distmat,
                cmc,
                g_camids,
                g_pids,
                mAP,
                q_camids,
                q_pids,
                eval_metric,
            )

        plot_body_parts_pairs_distance_distribution(
            body_parts_distmat, q_pids, g_pids, "Query-gallery"
        )
        print("Evaluate distribution of distances of pairs with same id vs different ids")
        (
            same_ids_dist_mean,
            same_ids_dist_std,
            different_ids_dist_mean,
            different_ids_dist_std,
            ssmd,
        ) = plot_pairs_distance_distribution(
            distmat, q_pids, g_pids, "Query-gallery"
        )
        print("Positive pairs distance distribution mean: {:.3f}".format(same_ids_dist_mean))
        print("Positive pairs distance distribution standard deviation: {:.3f}".format(same_ids_dist_std))
        print("Negative pairs distance distribution mean: {:.3f}".format(different_ids_dist_mean))
        print("Negative pairs distance distribution standard deviation: {:.3f}".format(different_ids_dist_std))
        print("SSMD = {:.4f}".format(ssmd))

        # if groundtruth target body masks are provided, compute part prediction accuracy
        avg_pxl_pred_accuracy = 0.0
        if (
                "target_masks" in q_anns
                and "target_masks" in g_anns
                and q_pxl_scores_ is not None
                and g_pxl_scores_ is not None
        ):
            q_pxl_pred_accuracy = self.compute_pixels_cls_accuracy(
                torch.from_numpy(q_anns["target_masks"]), q_pxl_scores_
            )
            g_pxl_pred_accuracy = self.compute_pixels_cls_accuracy(
                torch.from_numpy(g_anns["target_masks"]), g_pxl_scores_
            )
            avg_pxl_pred_accuracy = (
                                            q_pxl_pred_accuracy * len(q_parts_masks)
                                            + g_pxl_pred_accuracy * len(g_parts_masks)
                                    ) / (len(q_parts_masks) + len(g_parts_masks))
            print(
                "Pixel prediction accuracy for query = {:.2f}% and for gallery = {:.2f}% and on average = {:.2f}%".format(
                    q_pxl_pred_accuracy, g_pxl_pred_accuracy, avg_pxl_pred_accuracy
                )
            )

        if visrank:
            self.writer.visualize_rank(
                self.datamanager.test_loader[dataset_name],
                dataset_name,
                distmat,
                save_dir,
                visrank_topk,
                visrank_q_idx_list,
                visrank_count,
                body_parts_distmat,
                qf_parts_visibility,
                gf_parts_visibility,
                q_parts_masks,
                g_parts_masks,
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                q_anns,
                g_anns,
                eval_metrics,
            )

        self.writer.visualize_embeddings(
            qf,
            gf,
            q_pids,
            g_pids,
            self.datamanager.test_loader[dataset_name],
            dataset_name,
            qf_parts_visibility,
            gf_parts_visibility,
            mAP,
            cmc[0],
        )
        self.writer.performance_evaluation_timer.stop()
        return cmc, mAP, ssmd, avg_pxl_pred_accuracy

    def compute_pixels_cls_accuracy(self, target_masks, pixels_cls_scores):
        if pixels_cls_scores.is_cuda:
            target_masks = target_masks.cuda()
        # target_masks = nn.functional.interpolate(
        #     target_masks,
        #     pixels_cls_scores.shape[2::],
        #     mode="bilinear",
        #     align_corners=True,
        # )  # Best perf with bilinear here and nearest in resize transform
        pixels_cls_score_targets = target_masks.argmax(dim=1)  # [N, Hf, Wf]
        pixels_cls_score_targets = pixels_cls_score_targets.flatten()  # [N*Hf*Wf]
        pixels_cls_scores = pixels_cls_scores.permute(0, 2, 3, 1).flatten(
            0, 2
        )  # [N*Hf*Wf, M]
        accuracy = metrics.accuracy(pixels_cls_scores, pixels_cls_score_targets)[0]
        return accuracy.item()

    def display_individual_parts_ranking_performances(
        self,
        body_parts_distmat,
        cmc,
        g_camids,
        g_pids,
        mAP,
        q_camids,
        q_pids,
        eval_metric,
    ):
        print("Parts embeddings individual rankings :")
        bp_offset = 0
        if GLOBAL in self.config.model.kpr.test_embeddings:
            bp_offset += 1
        if FOREGROUND in self.config.model.kpr.test_embeddings:
            bp_offset += 1
        table = []
        for bp in range(
            0, body_parts_distmat.shape[0]
        ):  # TODO DO NOT TAKE INTO ACCOUNT -1 DISTANCES!!!!
            perf_metrics = metrics.evaluate_rank(
                body_parts_distmat[bp],
                q_pids,
                g_pids,
                q_camids,
                g_camids,
                eval_metric=eval_metric,
                max_rank=10,
                use_cython=False,
            )
            title = "p {}".format(bp - bp_offset)
            if bp < bp_offset:
                if bp == 0:
                    if GLOBAL in self.config.model.kpr.test_embeddings:
                        title = GLOBAL
                    else:
                        title = FOREGROUND
                if bp == 1:
                    title = FOREGROUND
            mAP = perf_metrics["mAP"]
            cmc = perf_metrics["cmc"]
            table.append([title, mAP, cmc[0], cmc[4], cmc[9]])
        headers = ["embed", "mAP", "R-1", "R-5", "R-10"]
        print(tabulate(table, headers, tablefmt="fancy_grid", floatfmt=".3f"))

    def parse_data_for_train(self, data):
        imgs = data["image"]
        imgs_path = data["img_path"]
        target_masks = data.get("target_masks", None)
        prompt_masks = data.get("prompt_masks", None)
        keypoints_xyc = data.get("keypoints_xyc", None)
        pids = data["pid"]
        cam_id = data["camid"]

        if self.use_gpu:
            imgs = imgs.cuda()
            cam_id = cam_id.cuda()
            if target_masks is not None:
                target_masks = target_masks.cuda()
            if prompt_masks is not None:
                prompt_masks = prompt_masks.cuda()
            if keypoints_xyc is not None:
                keypoints_xyc = keypoints_xyc.cuda()
            pids = pids.cuda()

        if target_masks is not None:
            assert target_masks.shape[1] == (
                self.config.model.kpr.masks.parts_num + 1
            ), f"masks.shape[1] ({target_masks.shape[1]}) != parts_num ({self.config.model.kpr.masks.parts_num + 1})"

        kp_quality = data.get("kp_quality", None)
        has_json = data.get("has_json", None)
        return imgs, target_masks, prompt_masks, keypoints_xyc, pids, imgs_path, cam_id, kp_quality, has_json

    def parse_data_for_eval(self, data):
        imgs = data["image"]
        target_masks = data.get("target_masks", None)
        prompt_masks = data.get("prompt_masks", None)
        keypoints_xyc = data.get("keypoints_xyc", None)
        pids = data["pid"]
        camids = data["camid"]
        return imgs, target_masks, prompt_masks, keypoints_xyc, pids, camids
