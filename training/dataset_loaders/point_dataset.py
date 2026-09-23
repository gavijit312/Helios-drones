from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class BuildingOcclusionDataset(Dataset):
    def __init__(self, data_dir: str, num_points: int = 2048):
        self.files = sorted(list(Path(data_dir).glob("*.npy")))
        self.num_points = num_points

        if len(self.files) == 0:
            raise RuntimeError(f"No .npy building crops found in {data_dir}. Run scripts/crop_building_clusters.py first.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        complete_pts = np.load(str(self.files[idx]))  # (N, 3)

        # Directional occlusion mask to simulate UAV blind spots
        view_dir = np.random.randn(3)
        view_dir /= np.linalg.norm(view_dir)

        projections = np.dot(complete_pts, view_dir)
        keep_mask = projections < np.percentile(projections, 75)
        partial_pts = complete_pts[keep_mask]

        if len(partial_pts) < self.num_points:
            indices = np.random.choice(len(partial_pts), self.num_points, replace=True)
        else:
            indices = np.random.choice(len(partial_pts), self.num_points, replace=False)
        partial_pts = partial_pts[indices]

        # Shape: (3, N) for 1D convolutions
        return {
            "partial": torch.from_numpy(partial_pts).float().permute(1, 0),
            "complete": torch.from_numpy(complete_pts).float().permute(1, 0),
        }