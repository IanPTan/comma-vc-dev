import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A helper block consisting of two 3x3 convolutions, each followed by BatchNorm and ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class Encoder(nn.Module):
    """
    Encoder model with 4 layers. Downsamples using PixelUnshuffle.
    Produces a bottleneck embedding representing regions of the image.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        
        # Layer 1: H, W
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.unshuffle1 = nn.PixelUnshuffle(downscale_factor=2)
        # after unshuffle1: channels = base_channels * 4
        self.proj1 = nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1)
        
        # Layer 2: H/2, W/2
        self.enc2 = ConvBlock(base_channels * 2, base_channels * 2)
        self.unshuffle2 = nn.PixelUnshuffle(downscale_factor=2)
        # after unshuffle2: channels = base_channels * 8
        self.proj2 = nn.Conv2d(base_channels * 8, base_channels * 4, kernel_size=1)
        
        # Layer 3: H/4, W/4
        self.enc3 = ConvBlock(base_channels * 4, base_channels * 4)
        self.unshuffle3 = nn.PixelUnshuffle(downscale_factor=2)
        # after unshuffle3: channels = base_channels * 16
        self.proj3 = nn.Conv2d(base_channels * 16, base_channels * 8, kernel_size=1)
        
        # Layer 4: H/8, W/8
        self.enc4 = ConvBlock(base_channels * 8, base_channels * 8)
        self.unshuffle4 = nn.PixelUnshuffle(downscale_factor=2)
        # after unshuffle4: channels = base_channels * 32
        self.proj4 = nn.Conv2d(base_channels * 32, base_channels * 16, kernel_size=1)
        
        # Bottleneck: H/16, W/16

    def forward(self, x):
        # Layer 1
        x1 = self.enc1(x)
        x = self.proj1(self.unshuffle1(x1))
        
        # Layer 2
        x2 = self.enc2(x)
        x = self.proj2(self.unshuffle2(x2))
        
        # Layer 3
        x3 = self.enc3(x)
        x = self.proj3(self.unshuffle3(x3))
        
        # Layer 4
        x4 = self.enc4(x)
        bottleneck = self.proj4(self.unshuffle4(x4))
        
        return bottleneck, [x1, x2, x3, x4]

class Decoder(nn.Module):
    """
    Decoder model with 4 layers. Upsamples using PixelShuffle.
    Uses skip connections from the encoder to reconstruct the image.
    """
    def __init__(self, out_channels=3, base_channels=32):
        super().__init__()
        
        # Input to decoder layer 4: bottleneck channels = base_channels * 16
        # PixelShuffle(2): channels = base_channels * 16 / 4 = base_channels * 4
        self.shuffle4 = nn.PixelShuffle(upscale_factor=2)
        # Cat skip4 (base_channels * 8) -> channels = base_channels * 12
        self.dec4 = ConvBlock(base_channels * 12, base_channels * 8)
        
        # PixelShuffle(2): channels = base_channels * 8 / 4 = base_channels * 2
        self.shuffle3 = nn.PixelShuffle(upscale_factor=2)
        # Cat skip3 (base_channels * 4) -> channels = base_channels * 6
        self.dec3 = ConvBlock(base_channels * 6, base_channels * 4)
        
        # PixelShuffle(2): channels = base_channels * 4 / 4 = base_channels
        self.shuffle2 = nn.PixelShuffle(upscale_factor=2)
        # Cat skip2 (base_channels * 2) -> channels = base_channels * 3
        self.dec2 = ConvBlock(base_channels * 3, base_channels * 2)
        
        # PixelShuffle(2): channels = base_channels * 2 / 4 = base_channels / 2
        self.shuffle1 = nn.PixelShuffle(upscale_factor=2)
        # Cat skip1 (base_channels) -> channels = base_channels * 1.5 (base_channels / 2 + base_channels)
        self.dec1 = ConvBlock(base_channels // 2 + base_channels, base_channels)
        
        # Final output layer to restore out_channels
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, bottleneck, skips):
        x1, x2, x3, x4 = skips
        
        # Layer 4
        x = self.shuffle4(bottleneck)
        x = torch.cat([x, x4], dim=1)
        x = self.dec4(x)
        
        # Layer 3
        x = self.shuffle3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        # Layer 2
        x = self.shuffle2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        # Layer 1
        x = self.shuffle1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        out = self.final_conv(x)
        return out

class Autoencoder(nn.Module):
    """Complete Autoencoder wrapping Encoder and Decoder modules."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(out_channels, base_channels)

    def forward(self, x):
        bottleneck, skips = self.encoder(x)
        out = self.decoder(bottleneck, skips)
        return out

if __name__ == '__main__':
    # Test dimensions (simulating frame sizes from data/0.mkv)
    original_height = 874
    original_width = 1164
    
    height = (original_height // 64) * 64
    width = (original_width // 64) * 64
    
    print(f"Original resolution: {original_width}x{original_height}")
    print(f"Resized resolution (multiple of 64): {width}x{height}")
    
    dummy_input = torch.randn(2, 3, height, width)
    print(f"Input shape: {dummy_input.shape}")
    
    encoder = Encoder()
    decoder = Decoder()
    
    bottleneck, skips = encoder(dummy_input)
    print(f"Bottleneck shape (embedding): {bottleneck.shape}")
    for i, skip in enumerate(skips, 1):
        print(f"Skip connection {i} shape: {skip.shape}")
        
    output = decoder(bottleneck, skips)
    print(f"Output shape: {output.shape}")
    
    # Also test the combined autoencoder
    ae = Autoencoder()
    ae_output = ae(dummy_input)
    print(f"Autoencoder output shape: {ae_output.shape}")
