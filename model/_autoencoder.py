import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    """
    A single layer of the Encoder.
    Applies PixelUnshuffle (halves H and W, quadruples channels) and then applies two Conv2d blocks.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(downscale_factor=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(self.unshuffle(x))

class DecoderLayer(nn.Module):
    """
    A single layer of the Decoder.
    Applies two Conv2d blocks and then applies PixelShuffle (doubles H and W, quarters channels).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * 4, out_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        return self.shuffle(self.conv(x))

class Encoder(nn.Module):
    """
    Encoder model with 4 layers. Downsamples using PixelUnshuffle.
    Produces a bottleneck embedding representing regions of the image.
    """
    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        self.layer1 = EncoderLayer(in_channels, base_channels)
        self.layer2 = EncoderLayer(base_channels, base_channels * 2)
        self.layer3 = EncoderLayer(base_channels * 2, base_channels * 4)
        self.layer4 = EncoderLayer(base_channels * 4, base_channels * 8)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        bottleneck = self.layer4(x)
        return bottleneck

class Decoder(nn.Module):
    """
    Decoder model with 4 layers. Upsamples using PixelShuffle.
    Reconstructs the image from the bottleneck embedding without any skip connections.
    """
    def __init__(self, out_channels=3, base_channels=32):
        super().__init__()
        self.layer4 = DecoderLayer(base_channels * 8, base_channels * 4)
        self.layer3 = DecoderLayer(base_channels * 4, base_channels * 2)
        self.layer2 = DecoderLayer(base_channels * 2, base_channels)
        self.layer1 = DecoderLayer(base_channels, out_channels)

    def forward(self, bottleneck):
        x = self.layer4(bottleneck)
        x = self.layer3(x)
        x = self.layer2(x)
        out = self.layer1(x)
        return out

class Autoencoder(nn.Module):
    """Complete Autoencoder wrapping Encoder and Decoder modules."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels)
        self.decoder = Decoder(out_channels, base_channels)

    def forward(self, x):
        bottleneck = self.encoder(x)
        out = self.decoder(bottleneck)
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
    
    bottleneck = encoder(dummy_input)
    print(f"Bottleneck shape (embedding): {bottleneck.shape}")
    
    output = decoder(bottleneck)
    print(f"Output shape: {output.shape}")
    
    # Also test the combined autoencoder
    ae = Autoencoder()
    ae_output = ae(dummy_input)
    print(f"Autoencoder output shape: {ae_output.shape}")
