# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# mae: https://github.com/facebookresearch/mae
# slowfast: https://github.com/facebookresearch/SlowFast
# --------------------------------------------------------

from functools import partial
from typing import Tuple, Optional, List

import math
import torch
import torch.nn as nn

from .hiera import Hiera, HieraBlock
from .hiera_utils import undo_windowing, conv_nd


class MaskedAutoencoderHiera(Hiera):
    """Masked Autoencoder with Hiera backbone (modified to use only a single final bottleneck)"""

    def __init__(
        self,
        in_chans: int = 3,
        patch_stride: Tuple[int, ...] = (4, 4),
        mlp_ratio: float = 4.0,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        **kwdargs,
    ):
        super().__init__(
            in_chans=in_chans,
            patch_stride=patch_stride,
            mlp_ratio=mlp_ratio,
            norm_layer=norm_layer,
            **kwdargs,
        )

        del self.norm, self.head
        encoder_dim_out = self.blocks[-1].dim_out
        self.encoder_norm = norm_layer(encoder_dim_out)
        self.mask_unit_spatial_shape_final = [
            i // s ** (self.q_pool) for i, s in zip(self.mask_unit_size, self.q_stride)
        ]
        self.tokens_spatial_shape_final = [
            i // s ** (self.q_pool)
            for i, s in zip(self.tokens_spatial_shape, self.q_stride)
        ]

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(encoder_dim_out, decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(
                1, math.prod(self.tokens_spatial_shape_final), decoder_embed_dim
            )
        )

        self.decoder_blocks = nn.ModuleList(
            [
                HieraBlock(
                    dim=decoder_embed_dim,
                    dim_out=decoder_embed_dim,
                    heads=decoder_num_heads,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                )
                for i in range(decoder_depth)
            ]
        )
        self.decoder_norm = norm_layer(decoder_embed_dim)

        self.pred_stride = patch_stride[-1] * (
            self.q_stride[-1] ** self.q_pool
        )  # patch stride of prediction

        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            (self.pred_stride ** min(2, len(self.q_stride))) * in_chans,
        )  # predictor
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._mae_init_weights)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def _mae_init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pixel_label_2d(
        self, input_img: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*
        input_img = input_img.permute(0, 2, 3, 1)

        size = self.pred_stride
        label = input_img.unfold(1, size, size).unfold(2, size, size)
        label = label.flatten(1, 2).flatten(2)
        label = label[mask]
        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def get_pixel_label_3d(
        self, input_vid: torch.Tensor, mask: torch.Tensor, norm: bool = True
    ) -> torch.Tensor:
        # mask (boolean tensor): True must correspond to *masked*

        # We use time strided loss, only take the first frame from each token
        input_vid = input_vid[:, :, ::self.patch_stride[0], :, :]

        size = self.pred_stride
        label = input_vid.unfold(3, size, size).unfold(4, size, size)
        label = label.permute(0, 2, 3, 4, 5, 6, 1)  # Different from 2d, mistake during training lol
        label = label.flatten(1, 3).flatten(2)
        label = label[mask]

        if norm:
            mean = label.mean(dim=-1, keepdim=True)
            var = label.var(dim=-1, keepdim=True)
            label = (label - mean) / (var + 1.0e-6) ** 0.5

        return label

    def forward_encoder(
        self, x: torch.Tensor, mask_ratio: float, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if mask is None:
            mask = self.get_random_mask(x, mask_ratio)  # [B, #MUs_all]

        # Get representation directly from final stage of Hiera encoder (no multiscale intermediates/fusion)
        x = super().forward(x, mask)
        
        # Roll it back to spatial order format for the decoder
        x = self.reroll(x, self.stage_ends[-1], mask=mask)

        x = self.encoder_norm(x)

        return x, mask

    def forward_decoder(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Embed tokens
        x = self.decoder_embed(x)

        # Combine visible and mask tokens

        # x: [B, #MUs, *mask_unit_spatial_shape_final, encoder_dim_out]
        # mask: [B, #MUs_all]
        x_dec = torch.zeros(*mask.shape, *x.shape[2:], device=x.device, dtype=x.dtype)
        mask_tokens = self.mask_token.view(
            (1,) * (len(mask.shape) + len(x.shape[2:-1])) + (-1,)
        )
        mask = mask.reshape(mask.shape + (1,) * len(x.shape[2:]))
        mask = mask.expand((-1,) * 2 + x.shape[2:]).bool()
        x_dec[mask] = x.flatten()
        x_dec = ~mask * mask_tokens + mask * x_dec

        # Get back spatial order
        x = undo_windowing(
            x_dec,
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )
        mask = undo_windowing(
            mask[..., 0:1],
            self.tokens_spatial_shape_final,
            self.mask_unit_spatial_shape_final,
        )

        # Flatten
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        mask = mask.view(x.shape[0], -1)

        # Add pos embed
        x = x + self.decoder_pos_embed

        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # Predictor projection
        x = self.decoder_pred(x)

        return x, mask

    def forward_loss(
        self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note: in mask, 0 is *visible*, 1 is *masked*

        x: e.g. [B, 3, H, W]
        pred: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        label: [B * num_pred_tokens, num_pixels_in_pred_patch * in_chans]
        """
        if len(self.q_stride) == 2:
            label = self.get_pixel_label_2d(x, mask)
        elif len(self.q_stride) == 3:
            label = self.get_pixel_label_3d(x, mask)
        else:
            raise NotImplementedError

        pred = pred[mask]
        loss = (pred - label) ** 2

        return loss.mean(), pred, label

    def forward(
        self,
        x: torch.Tensor,
        mask_ratio: float = 0.6,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        latent, mask = self.forward_encoder(x, mask_ratio, mask=mask)
        pred, pred_mask = self.forward_decoder(
            latent, mask
        )  # pred_mask is mask at resolution of *prediction*

        # Toggle mask, to generate labels for *masked* tokens
        return *self.forward_loss(x, pred, ~pred_mask), mask





if __name__ == "__main__":
    import torch
    print("Initializing small model...")
    # Create a small MAE Hiera model configured for high resolution to keep RAM/VRAM footprint small
    model = MaskedAutoencoderHiera(
        input_size=(896, 1152),
        patch_stride=(16, 16),       # Standard patch stride (yields 4032 tokens)
        embed_dim=64,                # Lower dim (divisible by 2 heads)
        num_heads=2,                 # Low non-one number of heads
        stages=(2, 2, 4, 2),         # Depth to test sequential block interactions
        q_pool=2,                    # Pool twice
        decoder_embed_dim=64,        # Decoder embedding dimension
        decoder_depth=2,             # Decoder depth
        decoder_num_heads=2          # Decoder heads
    )
    model.eval()
    
    # Input image: batch size 1, 3 channels, height 896, width 1152
    x = torch.randn(1, 3, 896, 1152)
    print(f"Input shape: {x.shape}")
    
    # 1. Test encoder
    print("\nTesting encoder...")
    latent, mask = model.forward_encoder(x, mask_ratio=0.6)
    print(f"Latent shape: {latent.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Check for NaNs/Infs in latent
    assert not torch.isnan(latent).any(), "Latent contains NaNs!"
    assert not torch.isinf(latent).any(), "Latent contains Infs!"
    print("Encoder checks passed: No NaNs or Infs found.")
    
    # 2. Test decoder
    print("\nTesting decoder...")
    pred, pred_mask = model.forward_decoder(latent, mask)
    print(f"Decoder prediction shape: {pred.shape}")
    print(f"Decoder prediction mask shape: {pred_mask.shape}")
    
    # Check for NaNs/Infs in prediction
    assert not torch.isnan(pred).any(), "Prediction contains NaNs!"
    assert not torch.isinf(pred).any(), "Prediction contains Infs!"
    print("Decoder checks passed: No NaNs or Infs found.")
    
    # 3. Test full forward loop
    print("\nTesting full forward pass...")
    loss, pred_loss, label_loss, final_mask = model(x, mask_ratio=0.6)
    print(f"Full forward loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN!"
    assert not torch.isinf(loss), "Loss is Inf!"
    print("Full forward checks passed: Loss is valid.")
    print("\nSMOKE TEST COMPLETED SUCCESSFULLY!")
