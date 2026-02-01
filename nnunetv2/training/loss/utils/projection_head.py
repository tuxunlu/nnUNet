# nnunetv2/training/loss/projection_head.py
import torch
import torch.nn as nn

class ProjectionHead2D(nn.Module):
    def __init__(self, in_ch: int, hid: int = 256, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hid, kernel_size=1, bias=False),
            nn.BatchNorm2d(hid), nn.ReLU(inplace=True),
            nn.Conv2d(hid, out, kernel_size=1, bias=True)
        )
    def forward(self, x): return self.net(x)

class ProjectionHead3D(nn.Module):
    def __init__(self, in_ch: int, hid: int = 256, out: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, hid, kernel_size=1, bias=False),
            nn.BatchNorm3d(hid), nn.ReLU(inplace=True),
            nn.Conv3d(hid, out, kernel_size=1, bias=True)
        )
    def forward(self, x): return self.net(x)
