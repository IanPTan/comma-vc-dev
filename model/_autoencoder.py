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
    Encoder model with configurable number of layers. Downsamples using PixelUnshuffle.
    Produces a bottleneck embedding representing regions of the image.
    """
    def __init__(self, in_channels=3, base_channels=32, num_layers=4, bottleneck_channels=256):
        super().__init__()
        self.layers = nn.ModuleList()
        curr_channels = in_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.layers.append(EncoderLayer(curr_channels, out_channels))
            curr_channels = out_channels
        
        if bottleneck_channels is not None:
            self.proj = nn.Conv2d(curr_channels, bottleneck_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.proj(x)

class Decoder(nn.Module):
    """
    Decoder model with configurable number of layers. Upsamples using PixelShuffle.
    Reconstructs the image from the bottleneck embedding without skip connections.
    """
    def __init__(self, out_channels=3, base_channels=32, num_layers=4, bottleneck_channels=256):
        super().__init__()
        
        enc_out_channels = base_channels * (2 ** (num_layers - 1))
        
        if bottleneck_channels is not None:
            self.proj = nn.Conv2d(bottleneck_channels, enc_out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
            
        self.layers = nn.ModuleList()
        curr_channels = enc_out_channels
        for i in reversed(range(num_layers)):
            if i > 0:
                layer_out_channels = base_channels * (2 ** (i - 1))
            else:
                layer_out_channels = out_channels
            self.layers.append(DecoderLayer(curr_channels, layer_out_channels))
            curr_channels = layer_out_channels

    def forward(self, bottleneck):
        x = self.proj(bottleneck)
        for layer in self.layers:
            x = layer(x)
        return x

class Autoencoder(nn.Module):
    """Complete Autoencoder wrapping Encoder and Decoder modules."""
    def __init__(self, in_channels=3, out_channels=3, base_channels=32, num_layers=4, bottleneck_channels=256):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, num_layers, bottleneck_channels)
        self.decoder = Decoder(out_channels, base_channels, num_layers, bottleneck_channels)

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
    
    encoder = Encoder(num_layers=4, bottleneck_channels=256)
    decoder = Decoder(num_layers=4, bottleneck_channels=256)
    
    bottleneck = encoder(dummy_input)
    print(f"Bottleneck shape (embedding): {bottleneck.shape}")
    
    output = decoder(bottleneck)
    print(f"Output shape: {output.shape}")
    
    # Also test the combined autoencoder with custom layers
    ae = Autoencoder(num_layers=3, base_channels=16, bottleneck_channels=128)
    ae_output = ae(dummy_input)
    print(f"Dynamic Autoencoder output shape: {ae_output.shape}")
