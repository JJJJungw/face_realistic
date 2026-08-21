"""Top-level FRS-v1 model."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attributes import TargetAttributeEncoder
from .config import V1ModelConfig
from .contracts import validate_v1_inputs
from .generator import CanonicalSwapGenerator
from .identity import MotionEncoder, MultiReferenceIdentityEncoder


class FaceRealisticSwapV1(nn.Module):
    """Generate an identity-swapped crop and masks without target RGB skips."""

    def __init__(self, config: V1ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or V1ModelConfig()
        self.identity_encoder = MultiReferenceIdentityEncoder(
            self.config.channels, self.config.identity_dim
        )
        self.motion_encoder = MotionEncoder(
            self.config.blendshape_dim + self.config.pose_dim,
            self.config.motion_embedding_dim,
        )
        self.attribute_encoder = TargetAttributeEncoder(
            self.config.geometry_channels, self.config.channels
        )
        self.generator = CanonicalSwapGenerator(
            self.config.channels,
            self.config.style_dim,
            self.config.attention_heads,
        )

    def forward(
        self,
        *,
        reference_faces: Tensor,
        geometry_maps: Tensor,
        blendshapes: Tensor,
        head_pose: Tensor,
        lowfreq_illumination: Tensor,
        context_ring: Tensor,
        occlusion_mask: Tensor,
        reference_quality: Tensor | None = None,
        reference_valid: Tensor | None = None,
    ) -> dict[str, Tensor]:
        validate_v1_inputs(
            config=self.config,
            reference_faces=reference_faces,
            geometry_maps=geometry_maps,
            blendshapes=blendshapes,
            head_pose=head_pose,
            lowfreq_illumination=lowfreq_illumination,
            context_ring=context_ring,
            occlusion_mask=occlusion_mask,
            reference_quality=reference_quality,
            reference_valid=reference_valid,
        )
        identity, local_identity, weights = self.identity_encoder(
            reference_faces, reference_quality, reference_valid
        )
        motion = self.motion_encoder(blendshapes, head_pose)
        attributes, sanitized_context, face_support = self.attribute_encoder(
            geometry_maps, lowfreq_illumination, context_ring, occlusion_mask
        )
        outputs = self.generator(
            attributes, local_identity, torch.cat((identity, motion), dim=1), face_support
        )
        alpha = outputs["raw_alpha"] * face_support
        visibility = outputs["raw_visibility"]
        composite_alpha = alpha * visibility * (1.0 - occlusion_mask.clamp(0.0, 1.0))
        return {
            **outputs,
            "alpha": alpha,
            "visibility": visibility,
            "composite_alpha": composite_alpha,
            "identity": identity,
            "reference_weights": weights,
            "sanitized_context": sanitized_context,
            "face_support": face_support,
        }
