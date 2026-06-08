import torch
import torch.nn as nn
import torch.nn.functional as F


#Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)  # skip connection


#Encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=3, hidden_dim=256, embed_dim=512):
    super().__init__()
    self.net = nn.Sequential(
        # 64x64x3 → 32x32x256
        nn.Conv2d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        ResidualBlock(hidden_dim),

        # 32x32x256 -> 16x16x256
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        ResidualBlock(hidden_dim),

        # 16x16x256 -> 8x8x256
        nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        ResidualBlock(hidden_dim),

        # 8x8x256 -> 8x8x512
        nn.Conv2d(hidden_dim, embed_dim, kernel_size=3, stride=1, padding=1),
        ResidualBlock(embed_dim),
    )

  def forward(self, x):
    return self.net(x)


#Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embed_dim=512, ema_decay=0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay

        # codebook vectors
        self.codebook = nn.Embedding(num_embeddings, embed_dim)
        self.codebook.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA tracking variables
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_embed_sum", self.codebook.weight.data.clone())

    def forward(self, z):
        # z: [B, D, H, W]
        z = z.permute(0, 2, 3, 1).contiguous()  # [B, H, W, D]
        flat_z = z.view(-1, self.embed_dim)       # [B*H*W, D]

        #find nearest codebook entry
        distances = (
            flat_z.pow(2).sum(dim=1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=1)
            - 2 * flat_z @ self.codebook.weight.t()
        )
        token_ids = distances.argmin(dim=1)  # [B*H*W]

        #look up quantized vectors
        quantized = self.codebook(token_ids).view(z.shape)

        # --- EMA codebook update ---
        if self.training:
            # One-hot encode the assignments
            encodings = F.one_hot(token_ids, self.num_embeddings).float()  # [B*H*W, K]

            # Update cluster sizes (how often each codebook entry is used)
            self.ema_cluster_size.data.mul_(self.ema_decay).add_(
                encodings.sum(0), alpha=1 - self.ema_decay
            )

            # Update embedding sums (sum of encoder outputs assigned to each entry)
            self.ema_embed_sum.data.mul_(self.ema_decay).add_(
                encodings.t() @ flat_z, alpha=1 - self.ema_decay
            )

            # Laplace smoothing to avoid division by zero
            n = self.ema_cluster_size.sum()
            cluster_size = (
                (self.ema_cluster_size + 1e-5)
                / (n + self.num_embeddings * 1e-5)
                * n
            )

            # Update codebook vectors
            self.codebook.weight.data.copy_(
                self.ema_embed_sum / cluster_size.unsqueeze(1)
            )

        # Commitment loss only (codebook is updated via EMA, not gradients)
        commitment_loss = F.mse_loss(z, quantized.detach())
        vq_loss = 0.25 * commitment_loss

        # Straight-through estimator
        quantized_st = z + (quantized - z).detach()
        quantized_st = quantized_st.permute(0, 3, 1, 2).contiguous()

        token_ids = token_ids.view(z.shape[0], z.shape[1], z.shape[2])

        return quantized_st, vq_loss, token_ids

    def reset_dead_entries(self, flat_z, max_reset=50):
        dead_mask = self.ema_cluster_size < 1.0
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        num_dead = len(dead_indices)
        if num_dead == 0:
            return 0
        num_to_reset = min(num_dead, max_reset)
        reset_indices = dead_indices[torch.randperm(num_dead, device=flat_z.device)[:num_to_reset]]
        n = flat_z.shape[0]
        replace_indices = torch.randint(0, n, (num_to_reset,), device=flat_z.device)
        self.codebook.weight.data[reset_indices] = flat_z[replace_indices]
        self.ema_embed_sum.data[reset_indices] = flat_z[replace_indices]
        self.ema_cluster_size.data[reset_indices] = 1.0
        return num_to_reset

    def codebook_usage(self):
        """Check how many codebook entries are actually being used."""
        used = (self.ema_cluster_size > 1.0).sum().item()
        return used, self.num_embeddings


#Decoder
class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_dim=256, embed_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            # 8x8x512 → 8x8x256
            nn.Conv2d(embed_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            ResidualBlock(hidden_dim),

            # 8x8x256 → 16x16x256
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dim),

            # 16x16x256 → 32x32x256
            nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(hidden_dim),

            # 32x32x256 → 64x64x3
            nn.ConvTranspose2d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        # x: [B, 512, 8, 8]
        # returns: [B, 3, 64, 64]
        return self.net(x)


# Full VQ-VAE
class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256, embed_dim=512,
                 num_embeddings=1024):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, embed_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embed_dim)
        self.decoder = Decoder(in_channels, hidden_dim, embed_dim)

    def forward(self, x):
        """
        Full forward pass:
            x → encode → quantize → decode → reconstruction

        Returns:
            recon: reconstructed frame [B, 3, 64, 64]
            vq_loss: vector quantization loss
            token_ids: discrete tokens [B, 8, 8]
        """
        z = self.encoder(x)
        quantized, vq_loss, token_ids = self.quantizer(z)
        recon = self.decoder(quantized)
        return recon, vq_loss, token_ids

    def encode(self, x):
        """Encode frames to discrete tokens (used after training)."""
        z = self.encoder(x)
        _, _, token_ids = self.quantizer(z)
        return token_ids  # [B, 8, 8]

    def decode_from_tokens(self, token_ids):
        """Decode discrete tokens back to frames (used for visualization)."""
        quantized = self.quantizer.codebook(token_ids)  # [B, 8, 8, 512]
        quantized = quantized.permute(0, 3, 1, 2)       # [B, 512, 8, 8]
        return self.decoder(quantized)                   # [B, 3, 64, 64]


# """
# Quick test to check the model is working.
# """

# if __name__ == "__main__":
#     model = VQVAE()
#     total_params = sum(p.numel() for p in model.parameters())
#     encoder_params = sum(p.numel() for p in model.encoder.parameters())
#     decoder_params = sum(p.numel() for p in model.decoder.parameters())
#     codebook_params = sum(p.numel() for p in model.quantizer.parameters())

#     print(f"Total parameters:    {total_params:,}")
#     print(f"  Encoder:           {encoder_params:,}")
#     print(f"  Codebook:          {codebook_params:,}")
#     print(f"  Decoder:           {decoder_params:,}")

#     # Fake batch
#     x = torch.randn(4, 3, 64, 64)
#     recon, vq_loss, tokens = model(x)

#     print(f"\nInput shape:          {x.shape}")
#     print(f"Reconstruction shape: {recon.shape}")
#     print(f"Token IDs shape:      {tokens.shape}")
#     print(f"Token ID range:       {tokens.min()} - {tokens.max()}")
#     print(f"VQ Loss:              {vq_loss.item():.4f}")
#     print(f"\nCompression: {64*64*3} values → {tokens.shape[1]*tokens.shape[2]} tokens ({64*64*3 / (tokens.shape[1]*tokens.shape[2]):.0f}x)")
#     print(f"Codebook: {model.quantizer.num_embeddings} visual words × {model.quantizer.embed_dim}-dim")

#     # Memory estimate for A100
#     param_mem = total_params * 4 / (1024**2)  # float32
#     print(f"\nModel memory: ~{param_mem:.0f} MB)")
