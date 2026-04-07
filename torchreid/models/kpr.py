from __future__ import division, absolute_import

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision.ops import FeaturePyramidNetwork

from torchreid import models
from torchreid.utils.constants import *

__all__ = [
    'kpr'
]



class KPR(nn.Module):
    """Keypoint Promptable Re-Identification model (KPR)."""
    def __init__(self, num_classes, pretrained, loss, config, horizontal_stripes=False, **kwargs):
        super(KPR, self).__init__()

        # Init config
        self.model_cfg = config.model.kpr
        self.num_classes = num_classes
        self.parts_num = self.model_cfg.masks.parts_num
        self.horizontal_stripes = horizontal_stripes
        self.shared_parts_id_classifier = self.model_cfg.shared_parts_id_classifier
        self.test_use_target_segmentation = self.model_cfg.test_use_target_segmentation
        self.training_binary_visibility_score = self.model_cfg.training_binary_visibility_score
        self.testing_binary_visibility_score = self.model_cfg.testing_binary_visibility_score
        self.use_prompt_visibility_score = self.model_cfg.use_prompt_visibility_score
        self.enable_fpn = self.model_cfg.enable_fpn
        self.fpn_out_dim = self.model_cfg.fpn_out_dim
        self.enable_msf = self.model_cfg.enable_msf

        # Backbone
        kwargs.pop("name", None)
        self.backbone_appearance_feature_extractor = models.build_model(
            self.model_cfg.backbone,
            num_classes,
            config=config,
            loss=loss,
            pretrained=pretrained,
            last_stride=self.model_cfg.last_stride,
            enable_dim_reduction=(self.model_cfg.dim_reduce=='before_pooling'),
            dim_reduction_channels=self.model_cfg.dim_reduce_output,
            pretrained_path=config.model.backbone_pretrained_path,
            use_as_backbone=True,
            enable_fpn=self.enable_msf or self.enable_fpn,
            **kwargs
        )
        self.spatial_feature_shape = self.backbone_appearance_feature_extractor.spatial_feature_shape
        self.spatial_feature_depth = self.spatial_feature_shape[2]

        # FPN + MSF
        if self.enable_fpn:
            out_channels = self.fpn_out_dim if not self.enable_msf else int(
                self.fpn_out_dim / len(self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer)
            )
            self.fpn = FeaturePyramidNetwork(
                self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer,
                out_channels=out_channels
            )
            self.spatial_feature_depth = out_channels
        if self.enable_msf:
            input_dim = self.fpn_out_dim if self.enable_fpn else self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer.sum()
            output_dim = self.backbone_appearance_feature_extractor.spatial_feature_depth_per_layer[-1]
            self.msf = MultiStageFusion(
                spatial_scale=self.model_cfg.msf_spatial_scale,
                img_size=(config.data.height, config.data.width),
                input_dim=input_dim,
                output_dim=output_dim
            )
            self.spatial_feature_depth = output_dim

        # Dim reduce
        self.init_dim_reduce_layers(self.model_cfg.dim_reduce,
                                    self.spatial_feature_depth,
                                    self.model_cfg.dim_reduce_output)

        # Pooling heads
        self.global_pooling_head = nn.AdaptiveAvgPool2d(1)
        self.foreground_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.background_attention_pooling_head = GlobalAveragePoolingHead(self.dim_reduce_output)
        self.parts_attention_pooling_head = init_part_attention_pooling_head(
            self.model_cfg.normalization,
            self.model_cfg.pooling,
            self.dim_reduce_output
        )

        # Part classifier
        self.learnable_attention_enabled = self.model_cfg.learnable_attention_enabled
        self.pixel_classifier = PixelToPartClassifier(self.spatial_feature_depth, self.parts_num)

        # ==============================
        # Identity classifier (CE 用)
        # ==============================
        num_train_ids = num_classes  # 保证和训练集 ID 数一致

        self.global_identity_classifier = BNClassifier(self.dim_reduce_output, num_train_ids)
        self.background_identity_classifier = BNClassifier(self.dim_reduce_output, num_train_ids)
        self.foreground_identity_classifier = BNClassifier(self.dim_reduce_output, num_train_ids)
        self.concat_parts_identity_classifier = BNClassifier(self.parts_num * self.dim_reduce_output, num_train_ids)
        if self.shared_parts_id_classifier:
            self.parts_identity_classifier = BNClassifier(self.dim_reduce_output, num_train_ids)
        else:
            self.parts_identity_classifier = nn.ModuleList(
                [BNClassifier(self.dim_reduce_output, num_train_ids) for _ in range(self.parts_num)]
            )

    def init_dim_reduce_layers(self, dim_reduce_mode, spatial_feature_depth, dim_reduce_output):
        self.dim_reduce_output = dim_reduce_output
        self.after_pooling_dim_reduce = False
        self.before_pooling_dim_reduce = None

        if dim_reduce_mode == 'before_pooling':
            self.before_pooling_dim_reduce = BeforePoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output)
            self.spatial_feature_depth = dim_reduce_output
        elif dim_reduce_mode in ['after_pooling', 'after_pooling_with_dropout',
                                 'before_and_after_pooling']:
            self.after_pooling_dim_reduce = True
            drop = 0.5 if dim_reduce_mode == 'after_pooling_with_dropout' else None
            self.global_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, drop)
            self.foreground_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, drop)
            self.background_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, drop)
            self.parts_after_pooling_dim_reduce = AfterPoolingDimReduceLayer(spatial_feature_depth, dim_reduce_output, drop)
        else:
            self.dim_reduce_output = spatial_feature_depth

    def forward(self, images, target_masks=None, prompt_masks=None, keypoints_xyc=None, cam_label=None, **kwargs):
        spatial_features = self.backbone_appearance_feature_extractor(
            images, prompt_masks=prompt_masks, keypoints_xyc=keypoints_xyc, cam_label=cam_label
        )

        if self.enable_fpn and isinstance(spatial_features, dict):
            spatial_features = self.fpn(spatial_features)
        if isinstance(spatial_features, dict):
            spatial_features = self.msf(spatial_features) if self.enable_msf else spatial_features[0]

        N, _, Hf, Wf = spatial_features.shape
        if self.before_pooling_dim_reduce is not None and spatial_features.shape[1] != self.dim_reduce_output:
            spatial_features = self.before_pooling_dim_reduce(spatial_features)

        # Pixel classification
        if self.horizontal_stripes:
            pixels_cls_scores = None
            feature_map_shape = (Hf, Wf)
            stripes_range = np.round(np.arange(0, self.parts_num + 1) * feature_map_shape[0] / self.parts_num).astype(int)
            pcb_masks = torch.zeros((self.parts_num, feature_map_shape[0], feature_map_shape[1]))
            for i in range(0, stripes_range.size - 1):
                pcb_masks[i, stripes_range[i]:stripes_range[i + 1], :] = 1
            pixels_parts_probabilities = pcb_masks
            pixels_parts_probabilities.requires_grad = False
            pixels_cls_scores = None
        elif self.learnable_attention_enabled:
            pixels_cls_scores = self.pixel_classifier(spatial_features)
            pixels_parts_probabilities = F.softmax(pixels_cls_scores, dim=1)
        else:
            assert target_masks is not None
            target_masks = target_masks.type(spatial_features.dtype)
            pixels_parts_probabilities = target_masks
            pixels_parts_probabilities.requires_grad = False
            pixels_cls_scores = None

        background_masks = pixels_parts_probabilities[:, 0]
        parts_masks = pixels_parts_probabilities[:, 1:]

        parts_visibility = parts_masks.amax(dim=(2, 3))
        visibility_scores = {PARTS: parts_visibility}

        # Global Embeddings
        global_embeddings = self.global_pooling_head(spatial_features)
        if global_embeddings.dim() > 2:
            global_embeddings = F.adaptive_avg_pool2d(global_embeddings, 1).view(N, -1)
        if self.after_pooling_dim_reduce:
            global_embeddings = self.global_after_pooling_dim_reduce(global_embeddings)
        bn_global_embeddings, global_cls_score = self.global_identity_classifier(global_embeddings)

        # Part Embeddings
        N, P, H, W = parts_masks.shape
        part_embeddings = []
        for i in range(P):
            mask = parts_masks[:, i].unsqueeze(1)
            masked_feat = spatial_features * mask
            pooled = F.adaptive_avg_pool2d(masked_feat, 1).view(N, -1)
            part_embeddings.append(pooled)
        part_embeddings = torch.stack(part_embeddings, dim=1)  # [N, P, D]

        # === 新增：统一降维 ===
        if self.after_pooling_dim_reduce:
            part_embeddings = self.parts_after_pooling_dim_reduce(part_embeddings)  # [N, P, 512]

        # Pack outputs
        embeddings = {
            GLOBAL: global_embeddings,
            BN_GLOBAL: bn_global_embeddings,
            PARTS: part_embeddings
        }
        id_cls_scores = {GLOBAL: global_cls_score}
        masks = {PARTS: parts_masks}

        if self.training:
            return embeddings, visibility_scores, parts_masks, pixels_cls_scores, spatial_features, masks, id_cls_scores
        else:
            return embeddings, visibility_scores, parts_masks, pixels_cls_scores, spatial_features, masks

    def parts_identity_classification(self, D, N, parts_embeddings):
        if self.shared_parts_id_classifier:
            parts_embeddings = parts_embeddings.flatten(0, 1)
            bn_part_embeddings, part_cls_score = self.parts_identity_classifier(parts_embeddings)
            bn_part_embeddings = bn_part_embeddings.view([N, self.parts_num, D])
            part_cls_score = part_cls_score.view([N, self.parts_num, -1])
        else:
            scores, embeddings = [], []
            for i, parts_identity_classifier in enumerate(self.parts_identity_classifier):
                bn_part_embeddings, part_cls_score = parts_identity_classifier(parts_embeddings[:, i])
                scores.append(part_cls_score.unsqueeze(1))
                embeddings.append(bn_part_embeddings.unsqueeze(1))
            part_cls_score = torch.cat(scores, 1)
            bn_part_embeddings = torch.cat(embeddings, 1)
        return bn_part_embeddings, part_cls_score




########################################
#    Dimensionality reduction layers   #
########################################

class MultiStageFusion(nn.Module):
    def __init__(self, input_dim, output_dim, mode="bilinear", img_size=None, spatial_scale=-1):
        super(MultiStageFusion, self).__init__()
        self.spatial_size = np.array(img_size) / spatial_scale if spatial_scale > 0 else None
        self.mode = mode
        self.dim_reduce = BeforePoolingDimReduceLayer(input_dim, output_dim)

    def forward(self, features_per_stage):
        spatial_size = self.spatial_size if self.spatial_size is not None else features_per_stage[0].size()[2:]
        resized_feature_maps = [features_per_stage[0]]
        for i in range(1, len(features_per_stage)):
            resized_feature_maps.append(
                F.interpolate(features_per_stage[i], size=spatial_size, mode=self.mode, align_corners=True)
            )
        fused_features = torch.cat(resized_feature_maps, 1)
        return self.dim_reduce(fused_features)


class BeforePoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BeforePoolingDimReduceLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, 1, stride=1, padding=0),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
        self._init_params()

    def forward(self, x): return self.layers(x)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01);
                if m.bias is not None: nn.init.constant_(m.bias, 0)


class AfterPoolingDimReduceLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p=None):
        super(AfterPoolingDimReduceLayer, self).__init__()
        layers = [
            nn.Linear(input_dim, output_dim, bias=True),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        ]
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))
        self.layers = nn.Sequential(*layers)
        self._init_params()

    def forward(self, x):
        if len(x.size()) == 3:
            N, K, _ = x.size()
            x = x.flatten(0, 1)
            x = self.layers(x).view(N, K, -1)
        else:
            x = self.layers(x)
        return x

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01);
                if m.bias is not None: nn.init.constant_(m.bias, 0)


########################################
#             Classifiers              #
########################################

class PixelToPartClassifier(nn.Module):
    def __init__(self, dim_reduce_output, parts_num):
        super(PixelToPartClassifier, self).__init__()
        self.bn = torch.nn.BatchNorm2d(dim_reduce_output)
        self.classifier = nn.Conv2d(in_channels=dim_reduce_output, out_channels=parts_num + 1, kernel_size=1)
        self._init_params()

    def forward(self, x): return self.classifier(self.bn(x))

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None: nn.init.constant_(m.bias, 0)


class BNClassifier(nn.Module):
    def __init__(self, in_dim, class_num):
        super(BNClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.bn.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_dim, class_num, bias=False)
        self._init_params()

    def forward(self, x):
        feat = self.bn(x)
        return feat, self.classifier(feat)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                if m.bias is not None: nn.init.constant_(m.bias, 0)


########################################
#            Pooling heads             #
########################################

def init_part_attention_pooling_head(normalization, pooling, dim_reduce_output):
    if pooling == 'gap':
        return GlobalAveragePoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gmp':
        return GlobalMaxPoolingHead(dim_reduce_output, normalization)
    elif pooling == 'gwap':
        return GlobalWeightedAveragePoolingHead(dim_reduce_output, normalization)
    else:
        raise ValueError('Unsupported pooling type {}'.format(pooling))


class GlobalMaskWeightedPoolingHead(nn.Module):
    def __init__(self, depth, normalization='identity'):
        super().__init__()
        if normalization == 'identity':
            self.normalization = nn.Identity()
        elif normalization == 'batch_norm_3d':
            self.normalization = torch.nn.BatchNorm3d(depth)
        elif normalization == 'batch_norm_2d':
            self.normalization = torch.nn.BatchNorm2d(depth)
        elif normalization == 'batch_norm_1d':
            self.normalization = torch.nn.BatchNorm1d(depth)
        else:
            raise ValueError('Unsupported normalization {}'.format(normalization))

    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)
        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = self.global_pooling(parts_features)
        return parts_features.view(N, M, -1)


class GlobalMaxPoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveMaxPool2d((1, 1))


class GlobalAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    global_pooling = nn.AdaptiveAvgPool2d((1, 1))


class GlobalWeightedAveragePoolingHead(GlobalMaskWeightedPoolingHead):
    def forward(self, features, part_masks):
        part_masks = torch.unsqueeze(part_masks, 2)
        features = torch.unsqueeze(features, 1)
        parts_features = torch.mul(part_masks, features)
        N, M, _, _, _ = parts_features.size()
        parts_features = parts_features.flatten(0, 1)
        parts_features = self.normalization(parts_features)
        parts_features = torch.sum(parts_features, dim=(-2, -1))
        part_masks_sum = torch.sum(part_masks.flatten(0, 1), dim=(-2, -1))
        part_masks_sum = torch.clamp(part_masks_sum, min=1e-6)
        return (parts_features / part_masks_sum).view(N, M, -1)


########################################
#             Constructors             #
########################################

def kpr(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    return KPR(num_classes, pretrained, loss, config, **kwargs)


def pcb(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    config.model.kpr.learnable_attention_enabled = False
    return KPR(num_classes, pretrained, loss, config, horizontal_stripes=True, **kwargs)


def bot(num_classes, loss='part_based', pretrained=True, config=None, **kwargs):
    config.model.kpr.masks.parts_num = 1
    config.model.kpr.learnable_attention_enabled = False
    return KPR(num_classes, pretrained, loss, config, horizontal_stripes=True, **kwargs)
