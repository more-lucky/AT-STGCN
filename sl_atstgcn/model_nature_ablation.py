"""Optional Nature-inspired skeleton ablation models.

This module is intentionally additive: the original ``sl_atstgcn.model`` module and
the original training results are untouched.  Use it only through
``scripts/train_nature_ablation.py`` and configs that set ``model_variant``.

Variants:
1. ``nature_1_feature_gate``: gated multi-source skeleton descriptor fusion.
2. ``nature_2_adaptive_temporal``: adaptive temporal branch selection.
3. ``nature_3_dual_graph``: joint graph plus feature-source graph gating.
"""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .graph import SKELETON_PARENT_JOINT_INDICES
from .keypoints import ALL_JOINTS, LEFT_SHOULDER_INDEX, MIDDLE_CHEST_INDEX, RIGHT_SHOULDER_INDEX, num_selected_joints
from .model import (
    MultiScaleGraphConv,
    MultiScaleTemporalConv,
    SpatialTemporalAttentionPool,
    SpatioTemporalChannelAttention,
    _cosine_margin_logits,
)


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


class MultiSourceGate(nn.Module):
    """Gate source descriptors before concatenation.

    The input sources are tensors with shape ``(N, C_s, T, V)``.  Each source is
    summarized with mean and standard deviation, optionally propagated over a
    learnable source graph, and converted to a scalar gate.  A non-zero minimum
    gate keeps this module conservative for ablation experiments.
    """

    def __init__(
        self,
        num_sources: int,
        *,
        hidden_dim: int = 24,
        min_gate: float = 0.20,
        use_source_graph: bool = False,
        source_graph_scale: float = 0.20,
        dynamic_source_graph: bool = True,
    ) -> None:
        super().__init__()
        self.num_sources = int(num_sources)
        self.min_gate = float(min_gate)
        self.use_source_graph = bool(use_source_graph)
        self.source_graph_scale = float(source_graph_scale)
        self.dynamic_source_graph = bool(dynamic_source_graph)
        hidden_dim = max(4, int(hidden_dim))
        self.embed = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        if self.use_source_graph:
            self.source_adjacency = nn.Parameter(torch.zeros(self.num_sources, self.num_sources))
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        else:
            self.register_parameter("source_adjacency", None)
            self.query = None
            self.key = None
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, parts: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        if len(parts) != self.num_sources:
            raise ValueError(f"Expected {self.num_sources} feature sources, got {len(parts)}")
        descriptors = []
        for _, tensor in parts:
            flat = tensor.flatten(1)
            descriptors.append(torch.stack([flat.mean(dim=1), flat.std(dim=1, unbiased=False)], dim=1))
        node_features = self.embed(torch.stack(descriptors, dim=1))
        if self.use_source_graph:
            identity = torch.eye(self.num_sources, device=node_features.device, dtype=node_features.dtype)
            static_adj = identity + torch.tanh(self.source_adjacency).to(node_features.dtype) * self.source_graph_scale
            static_adj = torch.softmax(static_adj, dim=-1)
            mixed = torch.einsum("ij,njh->nih", static_adj, node_features)
            if self.dynamic_source_graph and self.query is not None and self.key is not None:
                query = self.query(node_features)
                key = self.key(node_features)
                dynamic_adj = torch.softmax(
                    torch.matmul(query, key.transpose(1, 2)) / max(1.0, float(query.size(-1)) ** 0.5),
                    dim=-1,
                )
                mixed = mixed + torch.einsum("nij,njh->nih", dynamic_adj, node_features)
            node_features = mixed
        gates = torch.sigmoid(self.gate(node_features)).view(len(parts[0][1]), self.num_sources, 1, 1, 1)
        gates = self.min_gate + (1.0 - self.min_gate) * gates
        return [(name, tensor * gates[:, source_idx]) for source_idx, (name, tensor) in enumerate(parts)]


class AdaptiveMultiScaleTemporalConv(nn.Module):
    """Multi-scale temporal convolution with sample-adaptive branch weights."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1,),
        dropout: float = 0.0,
        gate_reduction: int = 8,
    ) -> None:
        super().__init__()
        channels = int(channels)
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
        branch_count = max(1, len(self.branches))
        hidden = max(4, channels // max(1, int(gate_reduction)))
        self.branch_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, branch_count, kernel_size=1, bias=True),
        )
        self.fuse = nn.Conv2d(channels * branch_count, channels, kernel_size=1, bias=False) if branch_count > 1 else nn.Identity()
        self.post = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0.0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pre(x)
        branch_outputs = [branch(x) for branch in self.branches]
        if len(branch_outputs) > 1:
            weights = torch.softmax(self.branch_gate(x).flatten(1), dim=1)
            weighted = [
                output * weights[:, idx].view(weights.size(0), 1, 1, 1)
                for idx, output in enumerate(branch_outputs)
            ]
            x = torch.cat(weighted, dim=1)
        else:
            x = branch_outputs[0]
        x = self.fuse(x)
        return self.post(x)


class NatureSkeletonGraphBlock(nn.Module):
    """Spatial graph convolution followed by original or adaptive temporal conv."""

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
        adaptive_temporal: bool = False,
        temporal_gate_reduction: int = 8,
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
        temporal_cls = AdaptiveMultiScaleTemporalConv if bool(adaptive_temporal) else MultiScaleTemporalConv
        temporal_kwargs = {
            "kernel_size": int(temporal_kernel),
            "dilations": tuple(temporal_dilations),
            "dropout": float(dropout),
        }
        if bool(adaptive_temporal):
            temporal_kwargs["gate_reduction"] = int(temporal_gate_reduction)
        self.tcn = temporal_cls(out_channels, **temporal_kwargs)
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        x = self.gcn(x)
        x = self.tcn(x)
        return self.activation(x + residual)


class NatureSkeletonSequenceFeatureEncoder(nn.Module):
    """Skeleton encoder with optional feature gating, adaptive TCN and source graph."""

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
        feature_gating: bool = False,
        feature_source_graph: bool = False,
        feature_gate_hidden_dim: int = 24,
        feature_gate_min: float = 0.20,
        source_graph_scale: float = 0.20,
        adaptive_temporal: bool = False,
        temporal_gate_reduction: int = 8,
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
        self.feature_gating = bool(feature_gating)
        self.feature_source_graph = bool(feature_source_graph)

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

        self.source_channels: list[tuple[str, int]] = [("joint", 2)]
        if self.use_bone_features:
            self.source_channels.append(("bone", 2))
        if self.use_motion_features:
            self.source_channels.append(("joint_motion", 2))
            if self.use_bone_features:
                self.source_channels.append(("bone_motion", 2))
        if self.include_acceleration:
            self.source_channels.append(("joint_acceleration", 2))
            if self.use_bone_features:
                self.source_channels.append(("bone_acceleration", 2))
        if self.include_absolute_xy:
            self.source_channels.append(("absolute_xy", 2))
        if self.include_root_motion:
            self.source_channels.append(("root_motion", 2))
        if self.include_validity:
            self.source_channels.append(("validity", 1))
        if self.include_temporal_position:
            self.source_channels.append(("temporal_position", 1))
        input_feature_channels = sum(channels for _, channels in self.source_channels)
        self.source_gate = None
        if self.feature_gating or self.feature_source_graph:
            self.source_gate = MultiSourceGate(
                len(self.source_channels),
                hidden_dim=int(feature_gate_hidden_dim),
                min_gate=float(feature_gate_min),
                use_source_graph=self.feature_source_graph,
                source_graph_scale=float(source_graph_scale),
            )
        self.input_bn = nn.BatchNorm2d(input_feature_channels)
        layers: list[nn.Module] = []
        in_channels = input_feature_channels
        for _ in range(max(1, int(blocks))):
            layers.append(
                NatureSkeletonGraphBlock(
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
                    adaptive_temporal=bool(adaptive_temporal),
                    temporal_gate_reduction=int(temporal_gate_reduction),
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
        parts: list[tuple[str, torch.Tensor]] = [("joint", joints)]
        if self.use_bone_features:
            parts.append(("bone", bones))
        if self.use_motion_features:
            joint_motion = self._temporal_delta(joints, valid)
            parts.append(("joint_motion", joint_motion))
            if self.use_bone_features:
                bone_motion = self._temporal_delta(bones, bone_valid)
                parts.append(("bone_motion", bone_motion))
        if self.include_acceleration:
            if joint_motion is None:
                joint_motion = self._temporal_delta(joints, valid)
            joint_acceleration = self._temporal_delta(joint_motion, valid)
            parts.append(("joint_acceleration", joint_acceleration))
            if self.use_bone_features:
                if bone_motion is None:
                    bone_motion = self._temporal_delta(bones, bone_valid)
                bone_acceleration = self._temporal_delta(bone_motion, bone_valid)
                parts.append(("bone_acceleration", bone_acceleration))
        if self.include_absolute_xy:
            absolute_xy = torch.where(valid[:, None], raw_xy.mul(2.0).sub(1.0), torch.zeros_like(raw_xy))
            parts.append(("absolute_xy", absolute_xy))
        if self.include_root_motion:
            root_xy = raw_xy[:, :, :, MIDDLE_CHEST_INDEX : MIDDLE_CHEST_INDEX + 1]
            root_valid = valid[:, :, MIDDLE_CHEST_INDEX : MIDDLE_CHEST_INDEX + 1]
            root_motion = self._temporal_delta(root_xy, root_valid).expand(-1, -1, -1, num_selected_joints())
            parts.append(("root_motion", root_motion))
        if self.include_validity:
            parts.append(("validity", valid[:, None].to(dtype=joints.dtype)))
        if self.include_temporal_position:
            positions = torch.linspace(-1.0, 1.0, x.size(2), device=x.device, dtype=x.dtype).view(1, 1, -1, 1)
            positions = positions.expand(x.size(0), 1, x.size(2), num_selected_joints())
            positions = torch.where(valid[:, None], positions, torch.zeros_like(positions))
            parts.append(("temporal_position", positions))
        if self.source_gate is not None:
            parts = self.source_gate(parts)
        features = torch.cat([tensor for _, tensor in parts], dim=1)
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


class NatureSkeletonOnlyClassifier(nn.Module):
    """Pure skeleton classifier using optional Nature-inspired modules."""

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
        feature_gating: bool = False,
        feature_source_graph: bool = False,
        feature_gate_hidden_dim: int = 24,
        feature_gate_min: float = 0.20,
        source_graph_scale: float = 0.20,
        adaptive_temporal: bool = False,
        temporal_gate_reduction: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = NatureSkeletonSequenceFeatureEncoder(
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
            feature_gating=feature_gating,
            feature_source_graph=feature_source_graph,
            feature_gate_hidden_dim=feature_gate_hidden_dim,
            feature_gate_min=feature_gate_min,
            source_graph_scale=source_graph_scale,
            adaptive_temporal=adaptive_temporal,
            temporal_gate_reduction=temporal_gate_reduction,
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


def build_nature_skeleton_classifier(
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
    feature_gating: bool = False,
    feature_source_graph: bool = False,
    feature_gate_hidden_dim: int = 24,
    feature_gate_min: float = 0.20,
    source_graph_scale: float = 0.20,
    adaptive_temporal: bool = False,
    temporal_gate_reduction: int = 8,
) -> NatureSkeletonOnlyClassifier:
    return NatureSkeletonOnlyClassifier(
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
        feature_gating=feature_gating,
        feature_source_graph=feature_source_graph,
        feature_gate_hidden_dim=feature_gate_hidden_dim,
        feature_gate_min=feature_gate_min,
        source_graph_scale=source_graph_scale,
        adaptive_temporal=adaptive_temporal,
        temporal_gate_reduction=temporal_gate_reduction,
    )


def build_nature_model_from_config(cfg: dict, *, num_classes: int, image_height: int) -> tuple[nn.Module, dict[str, Any]]:
    model_type = str(cfg.get("model_type", cfg.get("input_type", "skeleton"))).strip().lower()
    if model_type not in {"skeleton", "skeleton_only", "stgcn", "st-gcn"}:
        raise ValueError("Nature ablation models currently support skeleton input only.")
    variant = str(cfg.get("model_variant", "nature_1_feature_gate")).strip().lower()
    classifier_type = str(cfg.get("classifier_type", "linear"))
    logit_scale = float(cfg.get("logit_scale", 30.0))
    classifier_margin = float(cfg.get("classifier_margin", 0.0))
    center_loss_cfg = cfg.get("center_loss", {})
    if isinstance(center_loss_cfg, dict):
        center_loss_weight = float(center_loss_cfg.get("weight", 0.0 if not center_loss_cfg.get("enabled", False) else 0.02))
    else:
        center_loss_weight = 0.0
    sequence_length = int(cfg.get("sequence_length", image_height))
    skeleton_temporal_dilations = _parse_int_sequence(cfg.get("skeleton_temporal_dilations", (1,)))

    feature_gating = bool(cfg.get("nature_feature_gating", variant in {"nature_1_feature_gate", "1", "feature_gate"}))
    adaptive_temporal = bool(cfg.get("nature_adaptive_temporal", variant in {"nature_2_adaptive_temporal", "2", "adaptive_temporal"}))
    feature_source_graph = bool(cfg.get("nature_feature_source_graph", variant in {"nature_3_dual_graph", "3", "dual_graph"}))
    if feature_source_graph:
        feature_gating = True

    model = build_nature_skeleton_classifier(
        num_classes=num_classes,
        dropout=float(cfg["dropout"]),
        input_channels=int(cfg.get("input_channels", 3)),
        hidden_channels=int(cfg.get("skeleton_hidden_channels", cfg.get("hidden_channels", 192))),
        blocks=int(cfg.get("skeleton_blocks", cfg.get("blocks", 4))),
        adjacency_hops=int(cfg.get("skeleton_adjacency_hops", cfg.get("adjacency_hops", 2))),
        skeleton_dropout=float(cfg.get("skeleton_dropout", 0.10)),
        skeleton_temporal_kernel=int(cfg.get("skeleton_temporal_kernel", 5)),
        skeleton_temporal_dilations=skeleton_temporal_dilations,
        skeleton_adaptive_graph=bool(cfg.get("skeleton_adaptive_graph", False)),
        skeleton_edge_importance=bool(cfg.get("skeleton_edge_importance", False)),
        skeleton_adaptive_scale=float(cfg.get("skeleton_adaptive_scale", 0.10)),
        skeleton_relation_graph=bool(cfg.get("skeleton_relation_graph", False)),
        skeleton_relation_scale=float(cfg.get("skeleton_relation_scale", 0.05)),
        skeleton_relation_channels=int(cfg.get("skeleton_relation_channels", 32)),
        stc_attention=bool(cfg.get("skeleton_stc_attention", True)),
        center_joints=bool(cfg.get("skeleton_center_joints", True)),
        scale_normalize=bool(cfg.get("skeleton_scale_normalize", True)),
        hand_weight=float(cfg.get("skeleton_hand_weight", 1.20)),
        include_absolute_xy=bool(cfg.get("skeleton_include_absolute_xy", False)),
        include_validity=bool(cfg.get("skeleton_include_validity", False)),
        include_temporal_position=bool(cfg.get("skeleton_include_temporal_position", False)),
        include_root_motion=bool(cfg.get("skeleton_include_root_motion", False)),
        include_acceleration=bool(cfg.get("skeleton_include_acceleration", False)),
        use_bone_features=bool(cfg.get("skeleton_use_bone_features", True)),
        use_motion_features=bool(cfg.get("skeleton_use_motion_features", True)),
        pooling=str(cfg.get("skeleton_pooling", "avg")),
        part_pooling=bool(cfg.get("skeleton_part_pooling", False)),
        part_pooling_scale=float(cfg.get("skeleton_part_pooling_scale", 1.0)),
        classifier_type=classifier_type,
        logit_scale=logit_scale,
        classifier_margin=classifier_margin,
        center_loss_weight=center_loss_weight,
        feature_layer_norm=bool(cfg.get("feature_layer_norm", True)),
        feature_gating=feature_gating,
        feature_source_graph=feature_source_graph,
        feature_gate_hidden_dim=int(cfg.get("nature_feature_gate_hidden_dim", 24)),
        feature_gate_min=float(cfg.get("nature_feature_gate_min", 0.20)),
        source_graph_scale=float(cfg.get("nature_source_graph_scale", 0.20)),
        adaptive_temporal=adaptive_temporal,
        temporal_gate_reduction=int(cfg.get("nature_temporal_gate_reduction", 8)),
    )
    model_config = {
        "model_type": "skeleton",
        "model_variant": variant,
        "num_classes": num_classes,
        "image_height": sequence_length,
        "sequence_length": sequence_length,
        "input_channels": int(cfg.get("input_channels", 3)),
        "dropout": float(cfg["dropout"]),
        "classifier_type": classifier_type,
        "logit_scale": logit_scale,
        "classifier_margin": classifier_margin,
        "skeleton_hidden_channels": int(cfg.get("skeleton_hidden_channels", cfg.get("hidden_channels", 192))),
        "skeleton_blocks": int(cfg.get("skeleton_blocks", cfg.get("blocks", 4))),
        "skeleton_adjacency_hops": int(cfg.get("skeleton_adjacency_hops", cfg.get("adjacency_hops", 2))),
        "skeleton_dropout": float(cfg.get("skeleton_dropout", 0.10)),
        "skeleton_temporal_kernel": int(cfg.get("skeleton_temporal_kernel", 5)),
        "skeleton_temporal_dilations": list(skeleton_temporal_dilations),
        "skeleton_adaptive_graph": bool(cfg.get("skeleton_adaptive_graph", False)),
        "skeleton_edge_importance": bool(cfg.get("skeleton_edge_importance", False)),
        "skeleton_adaptive_scale": float(cfg.get("skeleton_adaptive_scale", 0.10)),
        "skeleton_relation_graph": bool(cfg.get("skeleton_relation_graph", False)),
        "skeleton_relation_scale": float(cfg.get("skeleton_relation_scale", 0.05)),
        "skeleton_relation_channels": int(cfg.get("skeleton_relation_channels", 32)),
        "skeleton_stc_attention": bool(cfg.get("skeleton_stc_attention", True)),
        "skeleton_center_joints": bool(cfg.get("skeleton_center_joints", True)),
        "skeleton_scale_normalize": bool(cfg.get("skeleton_scale_normalize", True)),
        "skeleton_hand_weight": float(cfg.get("skeleton_hand_weight", 1.20)),
        "skeleton_include_absolute_xy": bool(cfg.get("skeleton_include_absolute_xy", False)),
        "skeleton_include_validity": bool(cfg.get("skeleton_include_validity", False)),
        "skeleton_include_temporal_position": bool(cfg.get("skeleton_include_temporal_position", False)),
        "skeleton_include_root_motion": bool(cfg.get("skeleton_include_root_motion", False)),
        "skeleton_include_acceleration": bool(cfg.get("skeleton_include_acceleration", False)),
        "skeleton_use_bone_features": bool(cfg.get("skeleton_use_bone_features", True)),
        "skeleton_use_motion_features": bool(cfg.get("skeleton_use_motion_features", True)),
        "skeleton_pooling": str(cfg.get("skeleton_pooling", "avg")),
        "skeleton_part_pooling": bool(cfg.get("skeleton_part_pooling", False)),
        "skeleton_part_pooling_scale": float(cfg.get("skeleton_part_pooling_scale", 1.0)),
        "feature_layer_norm": bool(cfg.get("feature_layer_norm", True)),
        "center_loss_weight": center_loss_weight,
        "repair_missing_keypoints": bool(cfg.get("repair_missing_keypoints", False)),
        "repair_min_valid": int(cfg.get("repair_min_valid", 2)),
        "nature_feature_gating": feature_gating,
        "nature_adaptive_temporal": adaptive_temporal,
        "nature_feature_source_graph": feature_source_graph,
        "nature_feature_gate_hidden_dim": int(cfg.get("nature_feature_gate_hidden_dim", 24)),
        "nature_feature_gate_min": float(cfg.get("nature_feature_gate_min", 0.20)),
        "nature_source_graph_scale": float(cfg.get("nature_source_graph_scale", 0.20)),
        "nature_temporal_gate_reduction": int(cfg.get("nature_temporal_gate_reduction", 8)),
    }
    return model, model_config
