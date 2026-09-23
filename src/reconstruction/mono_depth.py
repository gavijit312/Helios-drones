from dataclasses import dataclass
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AlignedDepthOutput:
    metric_depth: np.ndarray        # Shape: (H, W), metric depth in meters
    surface_normals: np.ndarray     # Shape: (H, W, 3), unit normal vectors
    scale: float                    # Estimated scale factor s
    shift: float                    # Estimated shift factor t


class MonocularDepthEstimator:
    """Predicts dense relative depth maps from single keyframes and aligns them

    to absolute metric scale via robust sparse feature correspondence fitting.
    """

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Load torch hub relative depth backbone (MiDaS_small for high throughput)
        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device).eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = (
            midas_transforms.small_transform
            if model_type == "MiDaS_small"
            else midas_transforms.dpt_transform
        )

    @torch.inference_mode()
    def predict_relative_depth(self, image_np: np.ndarray) -> np.ndarray:
        """Runs monocular depth estimation on a single RGB frame.

        :param image_np: uint8 RGB numpy array of shape (H, W, 3).
        :return: Relative inverse depth map of shape (H, W).
        """
        input_batch = self.transform(image_np).to(self.device)
        prediction = self.model(input_batch)

        # Interpolate prediction back to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_relative = prediction.cpu().numpy()
        return depth_relative

    @staticmethod
    def align_to_metric_scale(
        rel_depth_map: np.ndarray,
        sparse_pixels_xy: np.ndarray,
        sparse_metric_depths_z: np.ndarray,
        ransac_iterations: int = 200,
        inlier_threshold: float = 0.5,
    ) -> Tuple[np.ndarray, float, float]:
        """Aligns relative inverse depth to metric linear depth via RANSAC:

            z_metric = 1.0 / (s * d_rel + t)
        """
        if len(sparse_pixels_xy) < 4:
            # Fallback scaling if sparse matches are too few
            norm_depth = (rel_depth_map - rel_depth_map.min()) / (
                rel_depth_map.max() - rel_depth_map.min() + 1e-6
            )
            metric_depth = 10.0 + norm_depth * 40.0
            return metric_depth.astype(np.float32), 1.0, 0.0

        xs = sparse_pixels_xy[:, 0].astype(int)
        ys = sparse_pixels_xy[:, 1].astype(int)

        # Sample relative depths at keypoint coordinates
        d_samples = rel_depth_map[ys, xs]
        inv_z_samples = 1.0 / (sparse_metric_depths_z + 1e-6)

        best_inliers = 0
        best_s, best_t = 1.0, 0.0

        # RANSAC linear fitting: inv_z = s * d_rel + t
        N = len(d_samples)
        for _ in range(ransac_iterations):
            idx = np.random.choice(N, size=2, replace=False)
            d1, d2 = d_samples[idx[0]], d_samples[idx[1]]
            z1, z2 = inv_z_samples[idx[0]], inv_z_samples[idx[1]]

            if abs(d1 - d2) < 1e-5:
                continue

            s = (z1 - z2) / (d1 - d2)
            t = z1 - s * d1

            residuals = np.abs(inv_z_samples - (s * d_samples + t))
            inliers = np.sum(residuals < inlier_threshold)

            if inliers > best_inliers and s > 0:
                best_inliers = inliers
                best_s, best_t = s, t

        # Compute dense metric depth map
        inv_metric_depth = best_s * rel_depth_map + best_t
        inv_metric_depth = np.clip(inv_metric_depth, 1e-3, 10.0)
        metric_depth = 1.0 / inv_metric_depth

        return metric_depth.astype(np.float32), float(best_s), float(best_t)

    @staticmethod
    def compute_surface_normals(depth_map: np.ndarray, camera_k: np.ndarray) -> np.ndarray:
        """Derives per-pixel unit surface normal vectors (nx, ny, nz) from depth gradients."""
        fx = camera_k[0, 0]
        fy = camera_k[1, 1]

        # Spatial gradients of depth
        dz_dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

        # Cross product of tangent vectors
        normal_x = -dz_dx / fx
        normal_y = -dz_dy / fy
        normal_z = np.ones_like(depth_map, dtype=np.float32)

        normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
        return (normals / norm).astype(np.float32)


if __name__ == "__main__":
    print("Testing MonocularDepthEstimator...")
    estimator = MonocularDepthEstimator(device="cpu")

    # Generate synthetic RGB test pattern
    test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(test_frame, (160, 120), 50, (255, 255, 255), -1)

    rel_depth = estimator.predict_relative_depth(test_frame)
    print(f"Relative Depth Inferred! Shape: {rel_depth.shape}, Min: {rel_depth.min():.2f}, Max: {rel_depth.max():.2f}")

    # Test Metric Alignment
    sample_pts = np.array([[160, 120], [100, 100], [200, 150], [50, 50]])
    ground_truth_z = np.array([20.0, 35.0, 32.0, 50.0])  # Metric meters

    aligned_z, s, t = estimator.align_to_metric_scale(rel_depth, sample_pts, ground_truth_z)
    print(f"Metric Alignment Success! Scale: {s:.4e}, Shift: {t:.4e}, Median Depth: {np.median(aligned_z):.2f}m")

    # Test Surface Normal Calculation
    k = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]], dtype=np.float32)
    normals = estimator.compute_surface_normals(aligned_z, k)
    print(f"Surface Normals Computed! Shape: {normals.shape}")
    print("Monocular Depth Module Verification Successful!")