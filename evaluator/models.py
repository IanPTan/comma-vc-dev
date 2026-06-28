#!/usr/bin/env python
import torch
import torch.nn as nn
import timm
import einops
import segmentation_models_pytorch as smp
from collections import namedtuple
from evaluator.utils import rgb_to_yuv6

Head = namedtuple('Head', ['name', 'hidden', 'out'])

BN_EPS = 0.001
BN_MOM = 0.01
VISION_FEATURES = 2048
SUMMARY_FEATURES = 512
IN_CHANS = 6 * 2
ACT_LAYER = 'gelu_tanh'
HEADS = [Head('pose', 32, 12)]

class AllNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = BN_EPS, momentum: float = BN_MOM, affine: bool = True):
        super().__init__()
        self.bn = nn.BatchNorm1d(1, eps, momentum, affine)
    def forward(self, x):
        return self.bn(x.view(-1, 1)).view(x.shape)

class ResBlock(nn.Module):
    def __init__(self, feats, expansion=2, norm=AllNorm):
        super().__init__()
        self.block_a = nn.Sequential(nn.Linear(feats, feats*expansion), norm(feats*expansion), nn.ReLU(inplace=True), nn.Linear(feats*expansion, feats), norm(feats))
        self.block_b = nn.Sequential(nn.ReLU(inplace=True), nn.Linear(feats, feats*expansion), norm(feats*expansion), nn.ReLU(inplace=True), nn.Linear(feats*expansion, feats), norm(feats))
        self.final_relu = nn.ReLU(inplace=False)
    def forward(self, x):
        a_out = x + self.block_a(x)
        return self.final_relu(a_out + self.block_b(a_out))

class Hydra(nn.Module):
    def __init__(self, num_features: int, heads: list[Head]=HEADS):
        super().__init__()
        self.resblock = ResBlock(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.heads = heads
        self.in_layer = nn.ModuleDict({k.name: nn.Linear(num_features, k.hidden) for k in heads})
        self.res_layer = nn.ModuleDict({h.name: nn.Sequential(nn.Linear(h.hidden, h.hidden), nn.ReLU(inplace=True), nn.Linear(h.hidden, h.hidden)) for h in heads})
        self.final_layer = nn.ModuleDict({h.name: nn.Linear(h.hidden, h.out) for h in heads})
    def forward(self, x):
        x = self.resblock(x)
        in_layer = {k: self.relu(v(x)) for k,v in self.in_layer.items()}
        res_layer = {k: self.relu(in_layer[k] + v(in_layer[k])) for k,v in self.res_layer.items()}
        ret = {k: v(res_layer[k]) for k,v in self.final_layer.items()}
        return ret

class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('_mean', torch.tensor([255 / 2] * IN_CHANS).view(1, IN_CHANS, 1, 1), persistent=True)
        self.register_buffer('_std', torch.tensor([255 / 4] * IN_CHANS).view(1, IN_CHANS, 1, 1), persistent=True)
        self.vision = timm.create_model('fastvit_t12', pretrained=False, num_classes=VISION_FEATURES, in_chans=IN_CHANS, act_layer=timm.layers.get_act_layer(ACT_LAYER))
        self.summarizer = nn.Sequential(nn.Linear(VISION_FEATURES, SUMMARY_FEATURES), nn.ReLU(inplace=True), ResBlock(SUMMARY_FEATURES))
        self.hydra = Hydra(num_features=SUMMARY_FEATURES, heads=HEADS)

    def preprocess_input(self, x):
        # x is (B, 2, C, H, W)
        batch_size, seq_len, *_ = x.shape
        x_flat = einops.rearrange(x, 'b t c h w -> (b t) c h w', b=batch_size, t=seq_len, c=3)
        # Interpolate to target input size (384, 512)
        x_interp = torch.nn.functional.interpolate(x_flat, size=(384, 512), mode='bilinear')
        yuv = rgb_to_yuv6(x_interp)
        return einops.rearrange(yuv, '(b t) c h w -> b (t c) h w', b=batch_size, t=seq_len, c=6)

    def forward(self, x):
        # x input here is the preprocessed (B, 12, 192, 256) tensor
        vision_out = self.vision((x - self._mean) / self._std)
        summary = self.summarizer(vision_out)
        return self.hydra(summary)

    def compute_distortion(self, out1, out2):
        distortion_heads = ['pose']
        return sum(
            (out1[h.name][..., : h.out // 2] - out2[h.name][..., : h.out // 2])
            .pow(2)
            .mean(dim=tuple(range(1, out1[h.name].ndim)))
            for h in self.hydra.heads if h.name in distortion_heads
        )

class SegNet(smp.Unet):
    def __init__(self):
        super().__init__('tu-efficientnet_b2', classes=5, activation=None, encoder_weights=None)

    def preprocess_input(self, x):
        # x is (B, C, H, W)
        # Interpolate to target input size (384, 512)
        return torch.nn.functional.interpolate(x, size=(384, 512), mode='bilinear')

    def compute_distortion(self, out1, out2):
        diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
        return diff.mean(dim=tuple(range(1, diff.ndim)))

from pathlib import Path
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "model" / "weights"
DEFAULT_POSENET_WEIGHTS = DEFAULT_WEIGHTS_DIR / "posenet.safetensors"
DEFAULT_SEGNET_WEIGHTS = DEFAULT_WEIGHTS_DIR / "segnet.safetensors"

