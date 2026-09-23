from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd


@dataclass
class MatchResult:
    keypoints0: np.ndarray  # Shape: (M, 2) in (x, y) coordinates
    keypoints1: np.ndarray  # Shape: (M, 2) in (x, y) coordinates
    confidence: np.ndarray  # Shape: (M,) match confidence scores
    num_matches: int


class KeypointMatcher:
    """Extracts and matches geometric features between image pairs

    using SuperPoint keypoints and LightGlue attention-based matching.
    """

    def __init__(
        self,
        max_num_keypoints: int = 2048,
        detection_threshold: float = 0.005,
        filter_threshold: float = 0.1,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Initialize SuperPoint extractor
        self.extractor = (
            SuperPoint(
                max_num_keypoints=max_num_keypoints,
                detection_threshold=detection_threshold,
                nms_radius=4,
            )
            .eval()
            .to(self.device)
        )

        # Initialize LightGlue matcher configured for SuperPoint descriptors
        self.matcher = (
            LightGlue(
                features="superpoint",
                filter_threshold=filter_threshold,
                depth_confidence=-1,  # Full depth evaluation for stability
            )
            .eval()
            .to(self.device)
        )

    def _prepare_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """Ensures tensor is [1, 1, H, W] grayscale float32 in [0, 1] on target device."""
        img = image.to(self.device)
        if img.dim() == 3:  # [C, H, W]
            if img.shape[0] == 3:
                # Standard RGB to Grayscale coefficients
                weights = torch.tensor([0.299, 0.587, 0.114], device=self.device).view(3, 1, 1)
                img = (img * weights).sum(dim=0, keepdim=True)
            img = img.unsqueeze(0)  # [1, 1, H, W]
        elif img.dim() == 2:  # [H, W]
            img = img.unsqueeze(0).unsqueeze(0)
        return img

    @torch.inference_mode()
    def match_pair(
        self,
        image0_tensor: torch.Tensor,
        image1_tensor: torch.Tensor,
        mask0: Optional[np.ndarray] = None,
        mask1: Optional[np.ndarray] = None,
    ) -> MatchResult:
        """Matches features between two consecutive keyframes.

        Optional dynamic masks (1 = keep, 0 = ignore) drop moving features.
        """
        t0 = self._prepare_tensor(image0_tensor)
        t1 = self._prepare_tensor(image1_tensor)

        # Extract features
        feats0 = self.extractor({"image": t0})
        feats1 = self.extractor({"image": t1})

        # Match with LightGlue
        matches_dict = self.matcher({"image0": feats0, "image1": feats1})

        # Remove batch dimension
        feats0, feats1, matches_dict = rbd(feats0), rbd(feats1), rbd(matches_dict)

        kpts0 = feats0["keypoints"]  # [N, 2]
        kpts1 = feats1["keypoints"]  # [M, 2]
        matches = matches_dict["matches"]  # [K, 2] indices
        scores = matches_dict["scores"]  # [K]

        # Extract matched pairs
        pts0 = kpts0[matches[..., 0]].cpu().numpy()
        pts1 = kpts1[matches[..., 1]].cpu().numpy()
        conf = scores.cpu().numpy()

        # Filter out points falling inside dynamic object masks if provided
        if mask0 is not None or mask1 is not None:
            valid_indices = []
            for i, (p0, p1) in enumerate(zip(pts0, pts1)):
                x0, y0 = int(round(p0[0])), int(round(p0[1]))
                x1, y1 = int(round(p1[0])), int(round(p1[1]))

                keep = True
                if mask0 is not None:
                    if 0 <= y0 < mask0.shape[0] and 0 <= x0 < mask0.shape[1]:
                        if mask0[y0, x0] == 0:
                            keep = False
                if mask1 is not None and keep:
                    if 0 <= y1 < mask1.shape[0] and 0 <= x1 < mask1.shape[1]:
                        if mask1[y1, x1] == 0:
                            keep = False

                if keep:
                    valid_indices.append(i)

            if valid_indices:
                pts0 = pts0[valid_indices]
                pts1 = pts1[valid_indices]
                conf = conf[valid_indices]
            else:
                pts0 = np.empty((0, 2))
                pts1 = np.empty((0, 2))
                conf = np.empty((0,))

        return MatchResult(
            keypoints0=pts0,
            keypoints1=pts1,
            confidence=conf,
            num_matches=len(pts0),
        )


if __name__ == "__main__":
    print("Testing KeypointMatcher initialization with structured geometry...")
    matcher = KeypointMatcher(device="cpu")

    # Create a synthetic image with geometric shapes (corners, circles, rectangles)
    import cv2
    synthetic = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(synthetic, (40, 40), (140, 140), (255, 255, 255), -1)
    cv2.circle(synthetic, (220, 100), 40, (200, 200, 200), -1)
    cv2.line(synthetic, (30, 260), (270, 260), (255, 255, 255), 5)
    cv2.rectangle(synthetic, (180, 180), (250, 240), (150, 150, 150), -1)

    t0 = torch.from_numpy(synthetic).permute(2, 0, 1).float() / 255.0
    # Create shifted frame (simulate drone translation by 10 pixels right and down)
    shifted = np.roll(synthetic, shift=10, axis=1)
    shifted = np.roll(shifted, shift=10, axis=0)
    t1 = torch.from_numpy(shifted).permute(2, 0, 1).float() / 255.0

    result = matcher.match_pair(t0, t1)
    print(f"Extraction and matching successful! Total feature matches: {result.num_matches}")