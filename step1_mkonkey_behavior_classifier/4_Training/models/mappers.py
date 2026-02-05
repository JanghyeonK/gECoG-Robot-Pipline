import torch
import torch.nn as nn


# ===========================
# Depthwise Blocks
# ===========================
def _act(): return nn.GELU()

# 1) Dilated version: expand receptive field (maintain channel independence)
class DepthwiseDilatedBlock(nn.Module):
    """
    depthwise conv with dilation. Channel independence (groups=C) + dilated convolution to expand receptive field
    """
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2):
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, padding=pad,
            dilation=dilation, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

# 2) Pyramid version: parallel depthwise conv with different kernel sizes, then sum per-channel
class DepthwisePyramidBlock(nn.Module):
    """
    For each channel, parallel depthwise conv with {k in kernel_sizes} → sum → BN → GELU
    """
    def __init__(self, channels: int, kernel_sizes=(3, 5, 7)):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, channels,
                      kernel_size=k, padding=k//2, groups=channels, bias=False)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm2d(channels)
        self.act = _act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = 0
        for conv in self.branches:
            y = y + conv(x)    # Sum only at same spatial location per-channel (maintain independence)
        return self.act(self.bn(y))

class DepthwiseResBlock(nn.Module):
    """
    Depthwise residual block.
    Input/output channels same, maintain channel independence (groups=C).
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.act = nn.GELU()
        self.conv1 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            groups=channels, bias=True
        )
        self.conv2 = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            groups=channels, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = out + residual
        out = self.act(out)
        return out


class DepthwiseBlock(nn.Module):
    """
    Depthwise block (non-residual).
    Conv → BN → GELU, maintain channel independence (groups=C).
    """
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size, padding=padding,
            groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# ===========================
# Mappers
# ===========================
class DilatedCNNMapper(nn.Module):
    """
    (B,5,32,H,W) -> (B,160)
    여러 단계의 dilation을 순차로 적용하여 독립 채널마다 넓은 문맥 수용.
    """
    def __init__(self, n_bands: int = 5, n_channels: int = 32,
                 kernel_size: int = 3, dilations=(1, 2, 4, 2, 1)):
        super().__init__()
        C = n_bands * n_channels
        blocks = [DepthwiseDilatedBlock(C, kernel_size, d) for d in dilations]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nb, nc, H, W = x.shape
        x = x.reshape(B, nb*nc, H, W)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x

class PyramidCNNMapper(nn.Module):
    """
    (B,5,32,H,W) -> (B,160)
    멀티-스케일(3/5/7 등) 필터를 채널별 병렬로 적용해 국소/중장거리 패턴 동시 포착.
    """
    def __init__(self, n_bands: int = 5, n_channels: int = 32,
                 depth: int = 3, kernel_sizes=(3,5,7)):
        super().__init__()
        C = n_bands * n_channels
        self.blocks = nn.Sequential(*[DepthwisePyramidBlock(C, kernel_sizes) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nb, nc, H, W = x.shape
        x = x.reshape(B, nb*nc, H, W)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x


# 3) Shared-Weight(가중치 공유) 버전: 파라미터/메모리 절약
#    (맵 간 간섭은 없음. 동일한 소형 CNN을 160개 맵에 공유 적용)
class SharedWeightCNNMapper(nn.Module):
    """
    (B,5,32,H,W) -> (B,160)
    (B*160,1,H,W)로 펼친 뒤, 소형 CNN을 '공유 가중치'로 적용(독립 처리 + 공유).
    """
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(32, affine=True, track_running_stats=False),
            nn.GELU(),
        )
        self.layer = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True, track_running_stats=False),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True, track_running_stats=False),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nb, nc, H, W = x.shape
        x = x.reshape(B*nb*nc, 1, H, W)          # 맵별 독립 처리(가중치 공유)
        x = self.stem(x)
        x = self.layer(x)
        x = self.pool(x).flatten(1)              # (B*160, 128)
        x = self.drop(x)
        x = self.head(x)                         # (B*160, 1)
        x = x.view(B, nb*nc)                     # (B, 160)
        return x

class ResCNNMapper(nn.Module):
    """
    (B, 5, 32, H, W) -> (B, 160)
    Depthwise residual 기반: 채널 간 상호작용 없이 공간 합성 + 잔차.
    """
    def __init__(self, n_bands: int = 5, n_channels: int = 32, depth: int = 3):
        super().__init__()
        C = n_bands * n_channels
        self.blocks = nn.Sequential(*[DepthwiseResBlock(C) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nb, nc, H, W = x.shape
        x = x.reshape(B, nb * nc, H, W)           # (B, C, H, W), C=160
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        return x


class CNNMapper(nn.Module):
    """
    (B, 5, 32, H, W) -> (B, 160)
    Depthwise(=groups=C) 블록을 얕게 반복: 채널 간 완전 독립 경로.
    """
    def __init__(self, n_bands: int = 5, n_channels: int = 32,
                 depth: int = 5, kernel_size: int = 3):
        super().__init__()
        C = n_bands * n_channels
        self.blocks = nn.Sequential(*[DepthwiseBlock(C, kernel_size) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, nb, nc, H, W = x.shape
        x = x.reshape(B, nb * nc, H, W)           # (B, C, H, W)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (B, C)
        return x


# ===========================
# Builder
# ===========================
def build_mapper(name: str, n_bands: int = 5, n_channels: int = 32, **kwargs):
    key = name.lower()
    if key in ("rescnn",):
        return ResCNNMapper(n_bands=n_bands, n_channels=n_channels, **kwargs)
    if key in ("cnn",):
        return CNNMapper(n_bands=n_bands, n_channels=n_channels, **kwargs)
    if key in ("dilatedcnn"):
        return DilatedCNNMapper(n_bands=n_bands, n_channels=n_channels, **kwargs)
    if key in ("pyramidcnn"):
        return PyramidCNNMapper(n_bands=n_bands, n_channels=n_channels, **kwargs)
    if key in ("sharedcnn"):
        return SharedWeightCNNMapper(**kwargs)
    raise ValueError(f"Unknown mapper: {name}")