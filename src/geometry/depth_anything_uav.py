from pathlib import Path
import cv2
import numpy as np
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
            nn.Sigmoid(),
        )

    def forward(self, relative_depth_map: torch.Tensor) -> torch.Tensor:
        feat = self.block1(relative_depth_map)
        feat = self.block2(feat)
        norm_depth = self.head(feat)
        return norm_depth * self.max_depth


class DepthEstimatorPipeline(nn.Module):
    def __init__(self, adapter_checkpoint: str = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        # 1. Foundation Monocular Depth Backbone (MiDaS)
        print("[DepthEngine] Loading monocular depth backbone (MiDaS_small)...")
        self.base_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True
        ).to(self.device)
        self.base_model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self.transform = midas_transforms.small_transform

        # 2. Metric Adapter
        self.adapter = UAVMetricDepthAdapter(in_channels=1, max_depth=120.0).to(self.device)
        if adapter_checkpoint and Path(adapter_checkpoint).exists():
            self.adapter.load_state_dict(
                torch.load(adapter_checkpoint, map_location=self.device)
            )
            print(f"[DepthEngine] Loaded trained UAV metric adapter: {adapter_checkpoint}")
        self.adapter.eval()

    @torch.no_grad()
    def estimate_depth(self, frame_bgr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        prediction = self.base_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        # Normalize disparity map to [0, 1] relative inverse depth
        d_min = torch.amin(prediction, dim=(2, 3), keepdim=True)
        d_max = torch.amax(prediction, dim=(2, 3), keepdim=True)
        rel_norm = (prediction - d_min) / (d_max - d_min + 1e-6)

        # Invert so larger distance = higher value
        rel_depth = 1.0 - rel_norm

        # Scale using trained UAV metric adapter
        metric_depth = self.adapter(rel_depth)
        return metric_depth.squeeze().cpu().numpy()