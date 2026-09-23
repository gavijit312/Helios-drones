import argparse
import math
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import torch

from src.completion.completion_net import PointCompletionAutoEncoder
from src.geometry.depth_anything_uav import DepthEstimatorPipeline
from src.ingestion.interpolator import TelemetryInterpolator
from src.ingestion.srt_parser import DJISRTParser
from src.preprocessing.keyframe_selector import KeyframeSelector
from src.reconstruction.mesher import PoissonReconstructor


class VisualOdometryTracker:
    def __init__(self, fx: float, fy: float, cx: float, cy: float):
        self.K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        self.orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_gray = None
        self.prev_kps = None
        self.prev_des = None

    def estimate_pose_delta(self, frame_bgr: np.ndarray, scale_step: float = 0.8):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kps, des = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None or des is None or len(kps) < 40:
            self.prev_gray = gray
            self.prev_kps = kps
            self.prev_des = des
            return np.eye(3), np.zeros((3, 1))

        if self.prev_des is None:
            self.prev_gray = gray
            self.prev_kps = kps
            self.prev_des = des
            return np.eye(3), np.zeros((3, 1))

        matches = self.matcher.match(self.prev_des, des)
        matches = sorted(matches, key=lambda m: m.distance)

        pts1 = np.float32([self.prev_kps[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps[m.trainIdx].pt for m in matches])

        if len(pts1) < 15:
            return np.eye(3), np.zeros((3, 1))

        E, mask = cv2.findEssentialMat(
            pts2, pts1, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.2
        )

        if E is None or E.shape != (3, 3):
            return np.eye(3), np.zeros((3, 1))

        _, R, t, _ = cv2.recoverPose(E, pts2, pts1, self.K, mask=mask)

        self.prev_gray = gray
        self.prev_kps = kps
        self.prev_des = des

        return R, t * scale_step


def run_helios_pipeline(video_path: str, srt_path: str, max_keyframes: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print(f"   HELIOS 3D UAV RECONSTRUCTION PIPELINE ({device.upper()})")
    print("=" * 60 + "\n")

    # Clean up stale output model to avoid showing old cached results
    output_path = Path("output/model_georeferenced.ply")
    if output_path.exists():
        output_path.unlink()
        print("[Setup] Removed stale output mesh cache.")

    # 1. Telemetry Parsing
    parser = DJISRTParser()
    records = parser.parse_file(srt_path)
    interpolator = TelemetryInterpolator(records)
    print(f"[Stage 1] Loaded {len(records)} telemetry records.")

    # 2. Keyframe Triage & Intrinsic Modeling
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Standardize working resolution to 640 max bound
    scale = 640.0 / max(orig_w, orig_h)
    img_w = int(orig_w * scale)
    img_h = int(orig_h * scale)

    # Estimate field of view (GoPro / Urban UAV lens is ~85 deg HFOV)
    hfov_rad = math.radians(85.0)
    fx = (img_w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx
    cx, cy = img_w / 2.0, img_h / 2.0

    print(f"[Stage 2] Resolution: {orig_w}x{orig_h} -> Working: {img_w}x{img_h}")
    vo_tracker = VisualOdometryTracker(fx, fy, cx, cy)
    selector = KeyframeSelector(sharpness_thresh=25.0, min_motion_pixels=4.0)

    keyframes = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pts_ms = int((frame_idx / fps) * 1000)
        if selector.should_keep(frame):
            resized = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
            lat, lon, alt, pitch = interpolator.query_pose(pts_ms)
            keyframes.append((resized, alt, pitch))

        frame_idx += 1
        if len(keyframes) >= max_keyframes:
            break
    cap.release()
    print(f"[Stage 2] Selected {len(keyframes)} stable keyframes.")

    if len(keyframes) == 0:
        raise RuntimeError("No keyframes met sharpness criteria.")

    # 3. Initialize Monocular Depth Estimator
    depth_engine = DepthEstimatorPipeline(
        adapter_checkpoint="pretrained/checkpoints/uav_depth_adapter.pt",
        device=device,
    )

    # 4. Point Cloud Reprojection with Trajectory Integration
    global_points = []
    global_colors = []

    R_cum = np.eye(3)
    t_cum = np.zeros((3, 1))
    u, v = np.meshgrid(np.arange(img_w), np.arange(img_h))

    print("[Stage 4] Generating metric depth and unprojecting into 3D world space...")
    for idx, (frame, alt, pitch) in enumerate(keyframes):
        # Update camera trajectory
        R_step, t_step = vo_tracker.estimate_pose_delta(frame, scale_step=0.7)
        R_cum = R_cum @ R_step
        t_cum = t_cum + R_cum @ t_step

        # Predict geometric depth
        pred_depth = depth_engine.estimate_depth(frame, img_h, img_w)

        # Print diagnostics on first frame
        if idx == 0:
            print(
                f"[Stage 4 Debug] Frame 0 depth range: min={np.min(pred_depth):.2f}, "
                f"max={np.max(pred_depth):.2f}, mean={np.mean(pred_depth):.2f}"
            )

        # Adaptive thresholding: clip lowest 2% (near-lens noise) and highest 10% (sky)
        p2 = float(np.percentile(pred_depth, 2))
        p90 = float(np.percentile(pred_depth, 90))
        valid = (pred_depth >= max(0.5, p2)) & (pred_depth <= min(120.0, p90))

        if not np.any(valid):
            valid = pred_depth > 0.1

        z = pred_depth[valid]
        x = (u[valid] - cx) * z / fx
        y = (v[valid] - cy) * z / fy

        pts_cam = np.stack([x, y, z], axis=-1)

        # Transform from camera coordinates to accumulated world coordinates
        pts_world = (R_cum @ pts_cam.T).T + t_cum.reshape(1, 3)

        # Apply camera pitch orientation
        pitch_rad = math.radians(pitch)
        R_pitch = np.array([
            [1, 0, 0],
            [0, math.cos(pitch_rad), -math.sin(pitch_rad)],
            [0, math.sin(pitch_rad), math.cos(pitch_rad)],
        ])
        pts_world = (R_pitch @ pts_world.T).T
        pts_world[:, 2] += alt

        cols = (cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[valid] / 255.0).astype(np.float64)

        global_points.append(pts_world)
        global_colors.append(cols)

    total_pts_collected = sum(len(p) for p in global_points)
    if total_pts_collected == 0:
        raise RuntimeError("No 3D points were generated across keyframes. Check depth inference output.")

    pts_concat = np.concatenate(global_points, axis=0)
    cols_concat = np.concatenate(global_colors, axis=0)
    print(f"[Stage 4] Successfully generated {len(pts_concat):,} raw 3D points.")

    # 5. Outlier Filtering & Occlusion Completion
    pcd_raw = o3d.geometry.PointCloud()
    pcd_raw.points = o3d.utility.Vector3dVector(pts_concat)
    pcd_raw.colors = o3d.utility.Vector3dVector(cols_concat)

    print("[Stage 5] Cleaning depth noise via statistical outlier removal...")
    pcd_clean, _ = pcd_raw.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.0)
    pts_clean = np.asarray(pcd_clean.points)
    cols_clean = np.asarray(pcd_clean.colors)

    completion_model = PointCompletionAutoEncoder(num_points=2048).to(device)
    comp_ckpt = "pretrained/checkpoints/geometry_completion.pt"
    if Path(comp_ckpt).exists() and len(pts_clean) >= 2048:
        completion_model.load_state_dict(torch.load(comp_ckpt, map_location=device))
        print(f"[Stage 5] Infilling structural voids using: {comp_ckpt}")
        completion_model.eval()

        choice = np.random.choice(len(pts_clean), 2048, replace=False)
        patch = pts_clean[choice]
        centroid = np.mean(patch, axis=0)
        norm_patch = patch - centroid
        scale_val = np.max(np.sqrt(np.sum(norm_patch**2, axis=1))) + 1e-6
        inp = torch.from_numpy((norm_patch / scale_val).T).float().unsqueeze(0).to(device)

        with torch.no_grad():
            completed = completion_model(inp).squeeze(0).cpu().numpy().T
            completed = completed * scale_val + centroid
            completed_cols = np.tile(np.array([0.65, 0.65, 0.65]), (len(completed), 1))

            pts_clean = np.vstack([pts_clean, completed])
            cols_clean = np.vstack([cols_clean, completed_cols])

    # 6. Screened Poisson Surface Meshing
    pcd_final = o3d.geometry.PointCloud()
    pcd_final.points = o3d.utility.Vector3dVector(pts_clean)
    pcd_final.colors = o3d.utility.Vector3dVector(cols_clean)

    mesher = PoissonReconstructor(depth=8, density_quantile=0.06, voxel_size=0.15)
    mesher.reconstruct_surface(pcd_final, str(output_path))

    print("\n" + "=" * 60)
    print(" Pipeline complete: output/model_georeferenced.ply")
    print(" Run 'python scripts/visualize_mesh.py' or view via web/.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/zurich_flight.mp4")
    parser.add_argument("--srt", default="data/zurich_flight.srt")
    parser.add_argument("--keyframes", type=int, default=50)
    args = parser.parse_args()
    run_helios_pipeline(args.video, args.srt, max_keyframes=args.keyframes)