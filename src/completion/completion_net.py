import torch
import torch.nn as nn
import torch.nn.functional as F


class PointCompletionAutoEncoder(nn.Module):
    """
    Point cloud completion autoencoder that infills occluded facades
    and roof shadows typical of single-pass aerial drone trajectories.
    """
    def __init__(self, num_points: int = 2048):
        super().__init__()
        self.num_points = num_points

        # Encoder: Extracts local and global geometric point features
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)

        # Bottleneck latent representations
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1024)

        # Decoder: Inpaints complete structural geometry
        self.dec1 = nn.Linear(1024, 1024)
        self.dec2 = nn.Linear(1024, num_points * 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 3, N)
        b, _, _ = x.shape
        net = F.relu(self.bn1(self.conv1(x)))
        net = F.relu(self.bn2(self.conv2(net)))
        net = self.bn3(self.conv3(net))

        # Symmetrical pooling for permutation invariance
        global_feat = torch.max(net, dim=2, keepdim=False)[0]

        latent = F.relu(self.fc1(global_feat))
        latent = F.relu(self.fc2(latent))

        out = F.relu(self.dec1(latent))
        out = self.dec2(out)
        return out.view(b, 3, self.num_points)