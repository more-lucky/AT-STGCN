"""Model implementations for the AT-STGCN paper project.

The active paper path is the skeleton-only AT-STGCN classifier.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .graph import SKELETON_PARENT_JOINT_INDICES
from .keypoints import ALL_JOINTS, LEFT_SHOULDER_INDEX, MIDDLE_CHEST_INDEX, RIGHT_SHOULDER_INDEX, num_selected_joints


def _build_normalized_adjacency(*, hops: int = 2) -> torch.Tensor:
    """Build multi-hop normalized adjacency matrices for the 68-joint tree."""
    num_joints = num_selected_joints()
    adjacency = torch.eye(num_joints, dtype=torch.float32)
    for child, parent in enumerate(SKELETON_PARENT_JOINT_INDICES):
        child = int(child)
        parent = int(parent)
        adjacency[child, parent] = 1.0
        adjacency[parent, child] = 1.0
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    norm = degree.rsqrt()
    adjacency = norm[:, None] * adjacency * norm[None, :]

    matrices = [torch.eye(num_joints, dtype=torch.float32)]
    current = adjacency
    for _ in range(max(1, int(hops))):
        matrices.append(current)
        current = torch.matmul(current, adjacency).clamp(0.0, 1.0)
    return torch.stack(matrices, dim=0)


def _parse_int_sequence(value: Any, default: tuple[int, ...] = (1,)) -> tuple[int, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    parsed = tuple(max(1, int(item)) for item in values)
    return parsed or tuple(default)


def _cosine_margin_logits(
    features: torch.Tensor,
    weight: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    classifier_type: str,
    training: bool,
    margin: float,
    scale: float,
) -> torch.Tensor:
    cosine = F.linear(F.normalize(features, dim=1), F.normalize(weight, dim=1)).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7)
    resolved = str(classifier_type).strip().lower()
    if labels is not None and bool(training) and float(margin) > 0.0 and resolved in {"cosface", "arcface"}:
        labels = labels.view(-1, 1)
        target_cosine = cosine.gather(1, labels)
        if resolved == "arcface":
            margin = float(margin)
            sine = torch.sqrt((1.0 - target_cosine.square()).clamp_min(1.0e-7))
            phi = target_cosine * math.cos(margin) - sine * math.sin(margin)
            threshold = math.cos(math.pi - margin)
            correction = math.sin(math.pi - margin) * margin
            target_logits = torch.where(target_cosine > threshold, phi, target_cosine - correction)
        else:
            target_logits = target_cosine - float(margin)
        cosine = cosine.scatter(1, labels, target_logits)
    return cosine * float(scale)


class MultiScaleGraphConv(nn.Module):
    """Multi-hop graph convolution with optional CTR-style relation refinement."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        adjacency_hops: int = 2,
        adaptive: bool = False,
        edge_importance: bool = False,
        adaptive_scale: float = 0.10,
        relation_adaptive: bool = False,
        relation_scale: float = 0.05,
        relation_channels: int = 32,
    ) -> None:
        super().__init__()
        adjacency = _build_normalized_adjacency(hops=adjacency_hops)
        self.register_buffer("adjacency", adjacency)
        self.adaptive = bool(adaptive)
        self.edge_importance_enabled = bool(edge_importance)
        self.adaptive_scale = float(adaptive_scale)
        self.relation_adaptive = bool(relation_adaptive)
        self.relation_scale = float(relation_scale)
        if self.adaptive:
            self.adaptive_adjacency = nn.Parameter(torch.zeros_like(adjacency))
        else:
            self.register_parameter("adaptive_adjacency", None)
        if self.edge_importance_enabled:
            self.edge_importance = nn.Parameter(torch.ones_like(adjacency))
        else:
            self.register_parameter("edge_importance", None)
        if self.relation_adaptive:
            relation_hidden = max(8, min(int(relation_channels), max(8, int(in_channels) // 2)))
            self.relation_theta = nn.Conv2d(in_channels, relation_hidden, kernel_size=1, bias=False)
            self.relation_phi = nn.Conv2d(in_channels, relation_hidden, kernel_size=1, bias=False)
            self.relation_proj = nn.Conv2d(relation_hidden, adjacency.size(0), kernel_size=1, bias=False)
            nn.init.zeros_(self.relation_proj.weight)
        else:
            self.relation_theta = None
            self.relation_phi = None
            self.relation_proj = None
        self.proj = nn.Conv2d(in_channels * adjacency.size(0), out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = self.adjacency
        if self.adaptive_adjacency is not None:
            adjacency = adjacency + torch.tanh(self.adaptive_adjacency) * self.adaptive_scale
        if self.edge_importance is not None:
            adjacency = adjacency * self.edge_importance
        if self.relation_proj is not None and self.relation_theta is not None and self.relation_phi is not None:
            theta = self.relation_theta(x).mean(dim=2)
            phi = self.relation_phi(x).mean(dim=2)
            relation = theta.unsqueeze(-1) - phi.unsqueeze(-2)
            dynamic = torch.tanh(self.relation_proj(relation)) * self.relation_scale
            supports = [
                torch.einsum("nctv,nvw->nctw", x, adjacency[support_idx].unsqueeze(0) + dynamic[:, support_idx])
                for support_idx in range(adjacency.size(0))
            ]
        else:
            supports = [torch.einsum("nctv,vw->nctw", x, adj) for adj in adjacency]
        return self.proj(torch.cat(supports, dim=1))


class MultiScaleTemporalConv(nn.Module):
    """Multi-branch temporal convolution for short and long signing dynamics."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1,),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pre = nn.Sequential(nn.BatchNorm2d(channels), nn.GELU())
        branches = []
        for dilation in dilations:
            dilation = int(max(1, dilation))
            padding = (int(kernel_size) // 2) * dilation
            branches.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=(int(kernel_size), 1),
                    padding=(padding, 0),
                    dilation=(dilation, 1),
                    bias=False,
                )
            )
        self.branches = nn.ModuleList(branches)
        if len(self.branches) > 1:
            self.fuse = nn.Conv2d(channels * len(self.branches), channels, kernel_size=1, bias=False)
        else:
            self.fuse = nn.Identity()
        self.post = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        x = torch.cat([branch(x) for branch in self.branches], dim=1)
        x = self.fuse(x)
        return self.post(x)


class SpatioTemporalChannelAttention(nn.Module):
    """Lightweight STC attention inspired by skeleton SLR GCN papers."""

    def __init__(self, channels: int, *, reduction: int = 16, temporal_kernel: int = 7) -> None:
        super().__init__()
        hidden = max(1, int(channels) // int(reduction))
        padding = int(temporal_kernel) // 2
        self.channel = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )
        self.temporal = nn.Conv2d(1, 1, kernel_size=(int(temporal_kernel), 1), padding=(padding, 0), bias=False)
        self.spatial = nn.Conv2d(1, 1, kernel_size=(1, 7), padding=(0, 3), bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * torch.sigmoid(self.channel(x))
        summary = torch.mean(x, dim=1, keepdim=True)
        x = x * torch.sigmoid(self.temporal(summary))
        x = x * torch.sigmoid(self.spatial(summary))
        return x


class SpatialTemporalAttentionPool(nn.Module):
    """Attention pooling over time and joints for discriminative signing phases."""

    def __init__(self, channels: int, *, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.score = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, t, v = x.shape
        weights = torch.softmax(self.score(x).flatten(2), dim=-1).view(n, 1, t, v)
        return torch.sum(x * weights, dim=(2, 3), keepdim=True)


class SkeletonGraphBlock(nn.Module):
    """Spatial graph convolution followed by temporal convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        adjacency_hops: int = 2,
        temporal_kernel: int = 5,
        temporal_dilations: tuple[int, ...] = (1,),
        adaptive_graph: bool = False,
        edge_importance: bool = False,
        adaptive_scale: float = 0.10,
        relation_graph: bool = False,
        relation_scale: float = 0.05,
        relation_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.gcn = MultiScaleGraphConv(
            in_channels,
            out_channels,
            adjacency_hops=adjacency_hops,
            adaptive=adaptive_graph,
            edge_importance=edge_importance,
            adaptive_scale=adaptive_scale,
            relation_adaptive=relation_graph,
            relation_scale=relation_scale,
            relation_channels=relation_channels,
        )
        self.tcn = MultiScaleTemporalConv(
            out_channels,
            kernel_size=int(temporal_kernel),
            dilations=tuple(temporal_dilations),
            dropout=float(dropout),
        )
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels))
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.activation(x + residual)


class SkeletonSequenceFeatureEncoder(nn.Module):
    """Encode compact skeleton tensors shaped ``(N, C, T, 68)`` directly."""

    def __init__(
        self,
        *,
        input_channels: int = 3,
        hidden_channels: int = 192,
        blocks: int = 4,
        adjacency_hops: int = 2,
        dropout: float = 0.10,
        temporal_kernel: int = 5,
        temporal_dilations: tuple[int, ...] = (1,),
        adaptive_graph: bool = False,
        edge_importance: bool = False,
        adaptive_scale: float = 0.10,
        relation_graph: bool = False,
        relation_scale: float = 0.05,
        relation_channels: int = 32,
        stc_attention: bool = True,
        center_joints: bool = True,
        scale_normalize: bool = True,
        hand_weight: float = 1.20,
        include_absolute_xy: bool = False,
        include_validity: bool = False,
        include_temporal_position: bool = False,
        include_root_motion: bool = False,
        include_acceleration: bool = False,
        use_bone_features: bool = True,
        use_motion_features: bool = True,
        pooling: str = "avg",
        part_pooling: bool = False,
        part_pooling_scale: float = 1.0,
    ) -> None:
        super().__init__()
        del input_channels
        self.center_joints = bool(center_joints)
        self.scale_normalize = bool(scale_normalize)
        self.include_absolute_xy = bool(include_absolute_xy)
        self.include_validity = bool(include_validity)
        self.include_temporal_position = bool(include_temporal_position)
        self.include_root_motion = bool(include_root_motion)
        self.include_acceleration = bool(include_acceleration)
        self.use_bone_features = bool(use_bone_features)
        self.use_motion_features = bool(use_motion_features)
        self.pooling = str(pooling).strip().lower()
        if self.pooling not in {"avg", "avgmax", "avgattn", "avgmaxattn"}:
            raise ValueError(f"Unknown skeleton pooling mode: {pooling!r}")
        self.use_attention_pooling = self.pooling in {"avgattn", "avgmaxattn"}
        self.part_pooling = bool(part_pooling)
        self.part_pooling_scale = float(part_pooling_scale)
        part_sources = (
            {"pose", "computed"},
            {"face"},
            {"left_hand"},
            {"right_hand"},
        )
        self.part_count = 0
        if self.part_pooling:
            for sources in part_sources:
                indices = [idx for idx, joint in enumerate(ALL_JOINTS) if joint.source in sources]
                if indices:
                    self.register_buffer(
                        f"part_indices_{self.part_count}",
                        torch.as_tensor(indices, dtype=torch.long),
                        persistent=False,
                    )
                    self.part_count += 1
        self.register_buffer(
            "parent_indices",
            torch.as_tensor(SKELETON_PARENT_JOINT_INDICES, dtype=torch.long),
            persistent=False,
        )
        joint_weights = [
            float(hand_weight) if joint.source in {"left_hand", "right_hand"} else 1.0
            for joint in ALL_JOINTS
        ]
        self.register_buffer("joint_weights", torch.as_tensor(joint_weights, dtype=torch.float32), persistent=False)
        input_feature_channels = 2
        if self.use_bone_features:
            input_feature_channels += 2
        if self.use_motion_features:
            input_feature_channels += 2
            if self.use_bone_features:
                input_feature_channels += 2
        if self.include_acceleration:
            input_feature_channels += 2
            if self.use_bone_features:
                input_feature_channels += 2
        if self.include_absolute_xy:
            input_feature_channels += 2
        if self.include_validity:
            input_feature_channels += 1
        if self.include_temporal_position:
            input_feature_channels += 1
        if self.include_root_motion:
            input_feature_channels += 2
        self.input_bn = nn.BatchNorm2d(input_feature_channels)
        layers: list[nn.Module] = []
        in_channels = input_feature_channels
        for _ in range(max(1, int(blocks))):
            layers.append(
                SkeletonGraphBlock(
                    in_channels,
                    int(hidden_channels),
                    adjacency_hops=int(adjacency_hops),
                    temporal_kernel=int(temporal_kernel),
                    temporal_dilations=tuple(temporal_dilations),
                    adaptive_graph=bool(adaptive_graph),
                    edge_importance=bool(edge_importance),
                    adaptive_scale=float(adaptive_scale),
                    relation_graph=bool(relation_graph),
                    relation_scale=float(relation_scale),
                    relation_channels=int(relation_channels),
                    dropout=float(dropout),
                )
            )
            in_channels = int(hidden_channels)
        if stc_attention:
            layers.append(SpatioTemporalChannelAttention(int(hidden_channels), reduction=16, temporal_kernel=7))
        self.blocks = nn.Sequential(*layers)
        self.attention_pool = SpatialTemporalAttentionPool(int(hidden_channels)) if self.use_attention_pooling else None
        pool_multiplier = 1
        if self.pooling in {"avgmax", "avgmaxattn"}:
            pool_multiplier += 1
        if self.use_attention_pooling:
            pool_multiplier += 1
        pool_regions = 1 + (self.part_count if self.part_pooling else 0)
        self.out_channels = int(hidden_channels) * pool_multiplier * pool_regions

    def _joint_xy(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.size(3) != num_selected_joints():
            raise ValueError(f"Expected skeleton tensor (N, C, T, 68), got {tuple(x.shape)}")
        raw_xy = x[:, :2]
        joints = raw_xy
        valid = (joints[:, 0] != 0.0) | (joints[:, 1] != 0.0)
        if self.center_joints:
            root = joints[:, :, :, MIDDLE_CHEST_INDEX : MIDDLE_CHEST_INDEX + 1]
            joints = torch.where(valid[:, None], joints - root, joints)
        if self.scale_normalize:
            left = joints[:, :, :, LEFT_SHOULDER_INDEX]
            right = joints[:, :, :, RIGHT_SHOULDER_INDEX]
            shoulder_distance = torch.linalg.vector_norm(left - right, dim=1).mean(dim=1).clamp_min(0.05)
            joints = joints / shoulder_distance[:, None, None, None]
        return joints, valid, raw_xy

    @staticmethod
    def _temporal_delta(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(values)
        if values.size(2) > 1:
            both_valid = valid[:, 1:] & valid[:, :-1]
            delta[:, :, 1:] = torch.where(
                both_valid[:, None],
                values[:, :, 1:] - values[:, :, :-1],
                torch.zeros_like(values[:, :, 1:]),
            )
        return delta

    def _pool_region(self, features: torch.Tensor) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool2d(features, (1, 1))]
        if self.pooling in {"avgmax", "avgmaxattn"}:
            pooled.append(F.adaptive_max_pool2d(features, (1, 1)))
        if self.attention_pool is not None:
            pooled.append(self.attention_pool(features))
        return torch.cat(pooled, dim=1) if len(pooled) > 1 else pooled[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        joints, valid, raw_xy = self._joint_xy(x)
        bones = None
        bone_valid = None
        if self.use_bone_features:
            parent_joints = joints.index_select(dim=3, index=self.parent_indices)
            parent_valid = valid.index_select(dim=2, index=self.parent_indices)
            bone_valid = valid & parent_valid
            bones = torch.where(bone_valid[:, None], joints - parent_joints, torch.zeros_like(joints))
        joint_motion = None
        bone_motion = None
        feature_parts = [joints]
        if self.use_bone_features:
            feature_parts.append(bones)
        if self.use_motion_features:
            joint_motion = self._temporal_delta(joints, valid)
            feature_parts.append(joint_motion)
            if self.use_bone_features:
                bone_motion = self._temporal_delta(bones, bone_valid)
                feature_parts.append(bone_motion)
        if self.include_acceleration:
            if joint_motion is None:
                joint_motion = self._temporal_delta(joints, valid)
            joint_acceleration = self._temporal_delta(joint_motion, valid)
            feature_parts.append(joint_acceleration)
            if self.use_bone_features:
                if bone_motion is None:
                    bone_motion = self._temporal_delta(bones, bone_valid)
                bone_acceleration = self._temporal_delta(bone_motion, bone_valid)
                feature_parts.append(bone_acceleration)
        if self.include_absolute_xy:
            absolute_xy = torch.where(valid[:, None], raw_xy.mul(2.0).sub(1.0), torch.zeros_like(raw_xy))
            feature_parts.append(absolute_xy)
        if self.include_root_motion:
            root_xy = raw_xy[:, :, :, MIDDLE_CHEST_INDEX : MIDDLE_CHEST_INDEX + 1]
            root_valid = valid[:, :, MIDDLE_CHEST_INDEX : MIDDLE_CHEST_INDEX + 1]
            root_motion = self._temporal_delta(root_xy, root_valid).expand(-1, -1, -1, num_selected_joints())
            feature_parts.append(root_motion)
        if self.include_validity:
            feature_parts.append(valid[:, None].to(dtype=joints.dtype))
        if self.include_temporal_position:
            positions = torch.linspace(-1.0, 1.0, x.size(2), device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
            positions = positions.expand(x.size(0), 1, x.size(2), num_selected_joints())
            positions = torch.where(valid[:, None], positions, torch.zeros_like(positions))
            feature_parts.append(positions)
        features = torch.cat(feature_parts, dim=1)
        features = features * self.joint_weights.view(1, 1, 1, -1)
        features = self.input_bn(features)
        features = self.blocks(features)
        pooled = [self._pool_region(features)]
        if self.part_pooling:
            for part_idx in range(self.part_count):
                indices = getattr(self, f"part_indices_{part_idx}")
                part_features = self._pool_region(features.index_select(dim=3, index=indices))
                pooled.append(part_features * self.part_pooling_scale)
        return torch.flatten(torch.cat(pooled, dim=1), 1)


class SkeletonOnlyClassifier(nn.Module):
    """Skeleton-only AT-STGCN classifier; no image CNN is used."""

    def __init__(
        self,
        *,
        num_classes: int,
        dropout: float = 0.3,
        input_channels: int = 3,
        hidden_channels: int = 192,
        blocks: int = 4,
        adjacency_hops: int = 2,
        skeleton_dropout: float = 0.10,
        skeleton_temporal_kernel: int = 5,
        skeleton_temporal_dilations: tuple[int, ...] = (1,),
        skeleton_adaptive_graph: bool = False,
        skeleton_edge_importance: bool = False,
        skeleton_adaptive_scale: float = 0.10,
        skeleton_relation_graph: bool = False,
        skeleton_relation_scale: float = 0.05,
        skeleton_relation_channels: int = 32,
        stc_attention: bool = True,
        center_joints: bool = True,
        scale_normalize: bool = True,
        hand_weight: float = 1.20,
        include_absolute_xy: bool = False,
        include_validity: bool = False,
        include_temporal_position: bool = False,
        include_root_motion: bool = False,
        include_acceleration: bool = False,
        use_bone_features: bool = True,
        use_motion_features: bool = True,
        pooling: str = "avg",
        part_pooling: bool = False,
        part_pooling_scale: float = 1.0,
        classifier_type: str = "linear",
        logit_scale: float = 30.0,
        classifier_margin: float = 0.0,
        center_loss_weight: float = 0.0,
        feature_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SkeletonSequenceFeatureEncoder(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            blocks=blocks,
            adjacency_hops=adjacency_hops,
            dropout=skeleton_dropout,
            temporal_kernel=skeleton_temporal_kernel,
            temporal_dilations=skeleton_temporal_dilations,
            adaptive_graph=skeleton_adaptive_graph,
            edge_importance=skeleton_edge_importance,
            adaptive_scale=skeleton_adaptive_scale,
            relation_graph=skeleton_relation_graph,
            relation_scale=skeleton_relation_scale,
            relation_channels=skeleton_relation_channels,
            stc_attention=stc_attention,
            center_joints=center_joints,
            scale_normalize=scale_normalize,
            hand_weight=hand_weight,
            include_absolute_xy=include_absolute_xy,
            include_validity=include_validity,
            include_temporal_position=include_temporal_position,
            include_root_motion=include_root_motion,
            include_acceleration=include_acceleration,
            use_bone_features=use_bone_features,
            use_motion_features=use_motion_features,
            pooling=pooling,
            part_pooling=part_pooling,
            part_pooling_scale=part_pooling_scale,
        )
        self.feature_norm = nn.LayerNorm(self.encoder.out_channels) if bool(feature_layer_norm) else nn.Identity()
        self.dropout = nn.Dropout(float(dropout))
        self.classifier_type = str(classifier_type).strip().lower()
        if self.classifier_type not in {"linear", "cosine", "cosface", "arcface"}:
            raise ValueError(f"Unknown classifier type: {classifier_type!r}")
        self.logit_scale = float(logit_scale)
        self.classifier_margin = float(classifier_margin)
        self.center_loss_weight = float(center_loss_weight)
        self.classifier = nn.Linear(
            self.encoder.out_channels,
            int(num_classes),
            bias=self.classifier_type == "linear",
        )
        if self.center_loss_weight > 0.0:
            self.class_centers = nn.Parameter(torch.empty(int(num_classes), self.encoder.out_channels))
            nn.init.normal_(self.class_centers, std=0.02)
        else:
            self.register_parameter("class_centers", None)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(self.encoder(x))

    def center_clustering_loss(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.class_centers is None or self.center_loss_weight <= 0.0:
            return features.new_zeros(())
        features = F.normalize(features, dim=1)
        centers = F.normalize(self.class_centers, dim=1)
        target_centers = centers.index_select(0, labels)
        loss = 1.0 - torch.sum(features * target_centers, dim=1)
        return loss.mean() * self.center_loss_weight

    def classify_features(self, x: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.classifier_type in {"cosine", "cosface", "arcface"}:
            return _cosine_margin_logits(
                x,
                self.classifier.weight,
                labels=labels,
                classifier_type=self.classifier_type,
                training=self.training,
                margin=self.classifier_margin,
                scale=self.logit_scale,
            )
        return self.classifier(x)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.extract_features(x)
        logits = self.classify_features(self.dropout(features), labels=labels)
        return (logits, features) if return_features else logits


def build_skeleton_classifier(
    *,
    num_classes: int,
    dropout: float = 0.3,
    input_channels: int = 3,
    hidden_channels: int = 192,
    blocks: int = 4,
    adjacency_hops: int = 2,
    skeleton_dropout: float = 0.10,
    skeleton_temporal_kernel: int = 5,
    skeleton_temporal_dilations: tuple[int, ...] = (1,),
    skeleton_adaptive_graph: bool = False,
    skeleton_edge_importance: bool = False,
    skeleton_adaptive_scale: float = 0.10,
    skeleton_relation_graph: bool = False,
    skeleton_relation_scale: float = 0.05,
    skeleton_relation_channels: int = 32,
    stc_attention: bool = True,
    center_joints: bool = True,
    scale_normalize: bool = True,
    hand_weight: float = 1.20,
    include_absolute_xy: bool = False,
    include_validity: bool = False,
    include_temporal_position: bool = False,
    include_root_motion: bool = False,
    include_acceleration: bool = False,
    use_bone_features: bool = True,
    use_motion_features: bool = True,
    pooling: str = "avg",
    part_pooling: bool = False,
    part_pooling_scale: float = 1.0,
    classifier_type: str = "linear",
    logit_scale: float = 30.0,
    classifier_margin: float = 0.0,
    center_loss_weight: float = 0.0,
    feature_layer_norm: bool = True,
) -> SkeletonOnlyClassifier:
    return SkeletonOnlyClassifier(
        num_classes=num_classes,
        dropout=dropout,
        input_channels=input_channels,
        hidden_channels=hidden_channels,
        blocks=blocks,
        adjacency_hops=adjacency_hops,
        skeleton_dropout=skeleton_dropout,
        skeleton_temporal_kernel=skeleton_temporal_kernel,
        skeleton_temporal_dilations=skeleton_temporal_dilations,
        skeleton_adaptive_graph=skeleton_adaptive_graph,
        skeleton_edge_importance=skeleton_edge_importance,
        skeleton_adaptive_scale=skeleton_adaptive_scale,
        skeleton_relation_graph=skeleton_relation_graph,
        skeleton_relation_scale=skeleton_relation_scale,
        skeleton_relation_channels=skeleton_relation_channels,
        stc_attention=stc_attention,
        center_joints=center_joints,
        scale_normalize=scale_normalize,
        hand_weight=hand_weight,
        include_absolute_xy=include_absolute_xy,
        include_validity=include_validity,
        include_temporal_position=include_temporal_position,
        include_root_motion=include_root_motion,
        include_acceleration=include_acceleration,
        use_bone_features=use_bone_features,
        use_motion_features=use_motion_features,
        pooling=pooling,
        part_pooling=part_pooling,
        part_pooling_scale=part_pooling_scale,
        classifier_type=classifier_type,
        logit_scale=logit_scale,
        classifier_margin=classifier_margin,
        center_loss_weight=center_loss_weight,
        feature_layer_norm=feature_layer_norm,
    )


def create_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.98,
    optimizer_name: str = "sgd",
    backbone_lr_scale: float = 1.0,
    attention_lr_scale: float = 1.0,
    classifier_lr_scale: float = 1.0,
    no_weight_decay_norm_bias: bool = False,
) -> torch.optim.Optimizer:
    param_groups: dict[tuple[float, float], dict] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lr_scale = 1.0
        if name.startswith("features."):
            lr_scale = float(backbone_lr_scale)
        elif (
            name.startswith("attention.")
            or name.startswith("encoder.")
            or name.startswith("skeleton_encoder.")
            or name.startswith("skeleton_adapter.")
            or name.startswith("cnn_fusion.")
            or name.startswith("skeleton_fusion.")
            or name.startswith("fusion_")
        ):
            lr_scale = float(attention_lr_scale)
        elif name.startswith("classifier.") or name.startswith("class_centers"):
            lr_scale = float(classifier_lr_scale)
        group_weight_decay = float(weight_decay)
        if no_weight_decay_norm_bias and (param.ndim <= 1 or name.endswith(".bias") or ".norm" in name.lower()):
            group_weight_decay = 0.0
        key = (float(lr_scale), group_weight_decay)
        if key not in param_groups:
            param_groups[key] = {
                "params": [],
                "lr": float(learning_rate) * float(lr_scale),
                "lr_scale": float(lr_scale),
                "weight_decay": group_weight_decay,
            }
        param_groups[key]["params"].append(param)
    groups = list(param_groups.values())
    resolved = str(optimizer_name).strip().lower()
    if resolved == "sgd":
        return torch.optim.SGD(
            groups,
            lr=float(learning_rate),
            momentum=float(momentum),
            nesterov=True,
        )
    if resolved == "adamw":
        return torch.optim.AdamW(groups, lr=float(learning_rate))
    raise ValueError(f"Unknown optimizer: {optimizer_name!r}")


def _infer_num_classes(state_dict: dict[str, torch.Tensor]) -> int:
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        raise ValueError("Cannot infer num_classes: checkpoint has no classifier.weight")
    return int(classifier_weight.shape[0])


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    model_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int | None = None,
    metrics: dict[str, float] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": dict(model_config),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = int(epoch)
    if metrics is not None:
        payload["metrics"] = {k: float(v) for k, v in metrics.items()}
    torch.save(payload, path)


def load_skeleton_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str | None = None,
    num_classes: int | None = None,
    image_height: int | None = None,
    dropout: float | None = None,
    normalize_input: bool | None = None,
    attention: str | None = None,
    feature_dropout: float | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a skeleton-only AT-STGCN checkpoint without downloading weights."""
    checkpoint = torch.load(path, map_location=device or "cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        config = dict(checkpoint.get("model_config", {}))
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
        config = {}
        checkpoint = {"model_state_dict": state_dict, "model_config": config}
    else:
        raise ValueError(f"Unsupported checkpoint format in {path}")

    resolved_num_classes = int(num_classes or config.get("num_classes") or _infer_num_classes(state_dict))
    resolved_model_type = str(config.get("model_type", "skeleton")).strip().lower()
    if resolved_model_type not in {"skeleton", "skeleton_only", "stgcn", "st-gcn", "at_stgcn", "at-stgcn"}:
        raise ValueError(f"Unsupported checkpoint model_type={resolved_model_type!r}; only skeleton AT-STGCN is supported.")
    model = build_skeleton_classifier(
        num_classes=resolved_num_classes,
        dropout=float(dropout if dropout is not None else config.get("dropout", 0.0)),
        input_channels=int(config.get("input_channels", 3)),
        hidden_channels=int(config.get("skeleton_hidden_channels", config.get("hidden_channels", 192))),
        blocks=int(config.get("skeleton_blocks", config.get("blocks", 4))),
        adjacency_hops=int(config.get("skeleton_adjacency_hops", config.get("adjacency_hops", 2))),
        skeleton_dropout=float(config.get("skeleton_dropout", 0.10)),
        skeleton_temporal_kernel=int(config.get("skeleton_temporal_kernel", 5)),
        skeleton_temporal_dilations=_parse_int_sequence(config.get("skeleton_temporal_dilations", (1,))),
        skeleton_adaptive_graph=bool(config.get("skeleton_adaptive_graph", False)),
        skeleton_edge_importance=bool(config.get("skeleton_edge_importance", False)),
        skeleton_adaptive_scale=float(config.get("skeleton_adaptive_scale", 0.10)),
        skeleton_relation_graph=bool(config.get("skeleton_relation_graph", False)),
        skeleton_relation_scale=float(config.get("skeleton_relation_scale", 0.05)),
        skeleton_relation_channels=int(config.get("skeleton_relation_channels", 32)),
        stc_attention=bool(config.get("skeleton_stc_attention", True)),
        center_joints=bool(config.get("skeleton_center_joints", True)),
        scale_normalize=bool(config.get("skeleton_scale_normalize", True)),
        hand_weight=float(config.get("skeleton_hand_weight", 1.20)),
        include_absolute_xy=bool(config.get("skeleton_include_absolute_xy", False)),
        include_validity=bool(config.get("skeleton_include_validity", False)),
        include_temporal_position=bool(config.get("skeleton_include_temporal_position", False)),
        include_root_motion=bool(config.get("skeleton_include_root_motion", False)),
        include_acceleration=bool(config.get("skeleton_include_acceleration", False)),
        use_bone_features=bool(config.get("skeleton_use_bone_features", True)),
        use_motion_features=bool(config.get("skeleton_use_motion_features", True)),
        pooling=str(config.get("skeleton_pooling", "avg")),
        part_pooling=bool(config.get("skeleton_part_pooling", False)),
        part_pooling_scale=float(config.get("skeleton_part_pooling_scale", 1.0)),
        classifier_type=str(config.get("classifier_type", "linear")),
        logit_scale=float(config.get("logit_scale", 30.0)),
        classifier_margin=float(config.get("classifier_margin", 0.0)),
        center_loss_weight=float(config.get("center_loss_weight", 0.0)),
        feature_layer_norm=bool(config.get("feature_layer_norm", True)),
    )
    model.load_state_dict(state_dict)
    if device is not None:
        model.to(device)
    return model, checkpoint
