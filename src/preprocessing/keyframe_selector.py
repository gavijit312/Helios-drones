import cv2
import numpy as np


class KeyframeSelector:
    def __init__(self, sharpness_thresh: float = 80.0, min_motion_pixels: float = 12.0):
        self.sharpness_thresh = sharpness_thresh
        self.min_motion_pixels = min_motion_pixels
        self.last_gray = None

    def calculate_sharpness(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def should_keep(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = self.calculate_sharpness(frame_bgr)

        if sharpness < self.sharpness_thresh:
            return False

        if self.last_gray is None:
            self.last_gray = gray
            return True

        # Sparse optical flow displacement
        flow = cv2.calcOpticalFlowFarneback(
            self.last_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_motion = np.mean(mag)

        if mean_motion >= self.min_motion_pixels:
            self.last_gray = gray
            return True

        return False