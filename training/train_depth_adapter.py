from pathlib import Path
import torch
import torch.nn as nn


class ResidualDepthBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        res = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class UAVMetricDepthAdapter(nn.Module):
    def __init__(self, in_channels: int = 1, max_depth: float = 120.0):
        super().__init__()
        self.max_depth = max_depth

        self.block1 = ResidualDepthBlock(in_channels, 64)
        self.block2 = ResidualDepthBlock(64, 32)
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, relative_depth_map: torch.Tensor) -> torch.Tensor:
        feat = self.block1(relative_depth_map)
        feat = self.block2(feat)
        norm_depth = self.head(feat)
        return norm_depth * self.max_depth


class DepthAnythingUAVPipeline(nn.Module):
    def __init__(self, adapter_checkpoint: str = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.adapter = UAVMetricDepthAdapter(in_channels=1, max_depth=120.0).to(self.device)

        if adapter_checkpoint and Path(adapter_checkpoint).exists():
            self.adapter.load_state_dict(
                torch.load(adapter_checkpoint, map_location=self.device)
            )
            print(f"Loaded metric adapter weights from: {adapter_checkpoint}")
        self.adapter.eval()

    @torch.no_grad()
    def estimate_metric_depth(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        relative_proxy = torch.mean(rgb_tensor, dim=1, keepdim=True)
        return self.adapter(relative_proxy)