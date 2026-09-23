from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn


class DepthEstimatorPipeline(nn.Module):
    def __init__(self, adapter_checkpoint: str = None, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)

        print("[DepthEngine] Loading MiDaS backbone...")
        self.base_model = torch.hub.load(
            "intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True
        ).to(self.device)
        self.base_model.eval()

        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        self.transform = midas_transforms.small_transform

    @torch.no_grad()
    def estimate_depth(
        self,
        frame_bgr: np.ndarray,
        target_h: int,
        target_w: int,
        z_near: float = 3.5,
        z_far: float = 24.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            depth_map: metric depth in meters
            valid_mask: boolean mask removing sky, horizon, and extreme sensor noise
        """
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)

        disparity = self.base_model(input_batch)
        disparity = torch.nn.functional.interpolate(
            disparity.unsqueeze(1),
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
        )

        disp = disparity.squeeze().cpu().numpy()

        # MiDaS disparity: high = near, low = far/sky
        # Discard the lowest 25% disparity (sky, horizon, infinity) to eliminate the comet tail
        disp_low = np.percentile(disp, 25)
        disp_high = np.percentile(disp, 98)

        valid_mask = (disp > disp_low) & (disp <= disp_high)

        # Normalize disparity strictly within the building/street range [0, 1]
        disp_clamped = np.clip(disp, disp_low, disp_high)
        disp_norm = (disp_clamped - disp_low) / (disp_high - disp_low + 1e-6)

        # Smooth affine depth mapping: disp_norm=1 -> z_near, disp_norm=0 -> z_far
        depth = z_near * z_far / (z_near + (z_far - z_near) * (1.0 - disp_norm))

        return depth.astype(np.float32), valid_mask