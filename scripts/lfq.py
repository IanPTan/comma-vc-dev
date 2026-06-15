"""Lookup-Free Quantization

Self-contained autoencoder + LFQ quantizer. The model takes video segments
of shape (B, C, T, H, W) directly — no per-frame flatten needed. Encoder/decoder
use 3D convs so frames inform each other temporally; T is preserved (no
temporal compression), spatial dims are compressed 8x.

Token grid shape: (T, H/8, W/8). With T=8, H=W=64, that's (8, 8, 8) tokens
per video segment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3d(in_c, out_c):
    """3D conv that halves H,W and keeps T. kernel=(3,4,4), stride=(1,2,2)."""
    return nn.Conv3d(in_c, out_c, kernel_size=(3, 4, 4),
                     stride=(1, 2, 2), padding=(1, 1, 1))


def _deconv3d(in_c, out_c):
    """3D transposed conv that doubles H,W and keeps T."""
    return nn.ConvTranspose3d(in_c, out_c, kernel_size=(3, 4, 4),
                              stride=(1, 2, 2), padding=(1, 1, 1))


class Encoder(nn.Module):
    """Compresses each spatial dim by 2**num_downsamples (T preserved).

    With num_downsamples=3 and 64x64 input: -> 8x8 latent (8x compression).
    With num_downsamples=4 and 1152x864 input: -> 72x54 latent (16x compression).
    """

    def __init__(self, in_channels=3, hidden_dim=128, embed_dim=64, num_downsamples=3):
        super().__init__()
        layers = []
        in_c = in_channels
        for _ in range(num_downsamples):
            layers.append(_conv3d(in_c, hidden_dim))
            layers.append(nn.SiLU())
            in_c = hidden_dim
        layers.append(nn.Conv3d(hidden_dim, embed_dim, kernel_size=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Mirror of Encoder; upsamples each spatial dim by 2**num_downsamples."""

    def __init__(self, out_channels=3, hidden_dim=128, embed_dim=64, num_downsamples=3):
        super().__init__()
        layers = [nn.Conv3d(embed_dim, hidden_dim, kernel_size=1)]
        for i in range(num_downsamples):
            layers.append(nn.SiLU())
            out_c = hidden_dim if i < num_downsamples - 1 else out_channels
            layers.append(_deconv3d(hidden_dim, out_c))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LFQ(nn.Module):
    """Lookup-Free Quantization.

    Each of `codebook_dim` latent dimensions is binarized to {-1, +1}.
    The implicit codebook is the full {-1, +1}^codebook_dim hypercube
    (size 2**codebook_dim), so there is no embedding table to look up.

    Loss = commitment_loss_weight * ||z - sign(z).detach()||^2
         + entropy_loss_weight   * (E[H(q|z)] - gamma * H(E[q|z]))

    where q(c|z) ∝ exp(2 z·c / temperature). The entropy term pulls per-sample
    assignments toward one-hot (low E[H]) while spreading usage across the
    codebook (high H(E)).

    Input shape: (B, D, *spatial), where *spatial is any number of trailing
    dims (e.g. (T, H, W) for video, (H, W) for images).
    """

    def __init__(
        self,
        embed_dim: int,
        codebook_dim: int = 14,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        diversity_gamma: float = 1.0,
        temperature: float = 1.0,
        usage_ema_decay: float = 0.99,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.codebook_dim = codebook_dim
        self.codebook_size = 2 ** codebook_dim
        self.entropy_loss_weight = entropy_loss_weight
        self.commitment_loss_weight = commitment_loss_weight
        self.diversity_gamma = diversity_gamma
        self.temperature = temperature
        self.usage_ema_decay = usage_ema_decay

        if embed_dim == codebook_dim:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        else:
            self.project_in = nn.Linear(embed_dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, embed_dim)

        # bit weights for converting a {0,1}^D binary code into an int token id
        bits_to_int = 2 ** torch.arange(codebook_dim - 1, -1, -1, dtype=torch.long)
        self.register_buffer("bits_to_int", bits_to_int, persistent=False)

        # Explicit enumeration of all 2^D codes in {-1, +1}^D for the entropy loss.
        # Fine up to ~2^16; above that, switch to the per-dim factorized approximation.
        ids = torch.arange(self.codebook_size, dtype=torch.long)
        bits = ((ids.unsqueeze(-1) & bits_to_int) > 0).float()
        self.register_buffer("codebook", bits * 2 - 1, persistent=False)

        # EMA usage counter, exposed via codebook_usage().
        self.register_buffer("code_usage", torch.zeros(self.codebook_size))

    def forward(self, z):
        # z: [B, embed_dim, *spatial]
        B = z.shape[0]
        spatial = z.shape[2:]

        # [B, embed_dim, *spatial] -> [B, *spatial, embed_dim] -> [N, embed_dim]
        z = z.movedim(1, -1).contiguous()
        flat_z = z.reshape(-1, self.embed_dim)
        flat_z = self.project_in(flat_z)  # [N, codebook_dim]

        # Independent per-dim binarization. sign(0) -> +1.
        quantized = torch.where(
            flat_z >= 0, flat_z.new_ones(()), -flat_z.new_ones(())
        )

        bits = (quantized > 0).long()
        token_ids = (bits * self.bits_to_int).sum(dim=-1)  # [N]

        # --- commitment loss ---
        commitment_loss = F.mse_loss(flat_z, quantized.detach())

        # --- entropy loss ---
        # For ±1 codes, ||c||^2 = D is constant, so softmax(-||z-c||^2/τ)
        # reduces to softmax(2 z·c / τ).
        logits = (2.0 / self.temperature) * (flat_z @ self.codebook.t())  # [N, 2^D]
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)

        per_sample_entropy = -(probs * log_probs).sum(dim=-1).mean()
        avg_probs = probs.mean(dim=0)
        batch_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        entropy_loss = per_sample_entropy - self.diversity_gamma * batch_entropy

        loss = (
            self.commitment_loss_weight * commitment_loss
            + self.entropy_loss_weight * entropy_loss
        )

        # --- straight-through estimator ---
        quantized_st = flat_z + (quantized - flat_z).detach()
        quantized_st = self.project_out(quantized_st)  # [N, embed_dim]
        quantized_st = quantized_st.view(B, *spatial, self.embed_dim).movedim(-1, 1).contiguous()

        if self.training:
            with torch.no_grad():
                batch_usage = torch.bincount(
                    token_ids, minlength=self.codebook_size
                ).to(self.code_usage.dtype)
                self.code_usage.mul_(self.usage_ema_decay).add_(
                    batch_usage, alpha=1 - self.usage_ema_decay
                )

        token_ids = token_ids.view(B, *spatial)
        return quantized_st, loss, token_ids

    def codebook_usage(self):
        used = (self.code_usage > 1e-3).sum().item()
        return used, self.codebook_size

    def reset_dead_entries(self, *args, **kwargs):
        # No-op for parity with VectorQuantizer's interface. Entropy loss
        # already pushes the model to use every code.
        return 0

    def decode_from_ids(self, token_ids):
        """[B, *spatial] long token ids -> [B, embed_dim, *spatial] features."""
        bits = (token_ids.unsqueeze(-1) & self.bits_to_int) > 0
        codes = bits.to(self.codebook.dtype) * 2 - 1
        out = self.project_out(codes)  # [B, *spatial, embed_dim]
        return out.movedim(-1, 1).contiguous()


class LFQVAE(nn.Module):
    """Video autoencoder with LFQ. Takes (B, C, T, H, W) directly."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        codebook_dim: int = 14,
        num_downsamples: int = 3,
        entropy_loss_weight: float = 0.1,
        commitment_loss_weight: float = 0.25,
        # accepted but unused; lets train.py's --num-embeddings flag pass through
        num_embeddings: int | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, embed_dim, num_downsamples)
        self.quantizer = LFQ(
            embed_dim=embed_dim,
            codebook_dim=codebook_dim,
            entropy_loss_weight=entropy_loss_weight,
            commitment_loss_weight=commitment_loss_weight,
        )
        self.decoder = Decoder(in_channels, hidden_dim, embed_dim, num_downsamples)

    def forward(self, x):
        # x: (B, C, T, H, W)
        z = self.encoder(x)
        q, q_loss, tokens = self.quantizer(z)
        recon = self.decoder(q)
        return recon, q_loss, tokens

    def encode(self, x):
        z = self.encoder(x)
        _, _, tokens = self.quantizer(z)
        return tokens  # (B, T, H/8, W/8)

    def decode_from_tokens(self, token_ids):
        # token_ids: (B, T, H/8, W/8)
        q = self.quantizer.decode_from_ids(token_ids)
        return self.decoder(q)


if __name__ == "__main__":
    model = LFQVAE(codebook_dim=14)
    x = torch.randn(2, 3, 8, 64, 64)  # B=2, C=3, T=8, H=W=64
    recon, loss, tokens = model(x)
    print(f"recon:  {tuple(recon.shape)}")
    print(f"tokens: {tuple(tokens.shape)}  range [{tokens.min().item()}, {tokens.max().item()}]")
    print(f"loss:   {loss.item():.4f}")
    print(f"codebook size: {model.quantizer.codebook_size}")
    print(f"params: {sum(p.numel() for p in model.parameters()):,}")
