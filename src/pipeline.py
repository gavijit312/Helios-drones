import argparse
import math
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d
import torch

from src.geometry.depth_anything_uav import DepthEstimatorPipeline
from src.ingestion.interpolator import TelemetryInterpolator
from src.ingestion.srt_parser import DJISRTParser


def run_helios_pipeline(video_path: str, srt_path: str, max_keyframes: int = 15):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 60)
    print(f"   HELIOS 3D UAV RECONSTRUCTION PIPELINE ({device.upper()})")
    print("=" * 60 + "\n")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_mesh_path = output_dir / "model_georeferenced.ply"
    output_pcd_path = output_dir / "debug_points.ply"

    # 1. Telemetry Ingestion
    parser = DJISRTParser()
    records = parser.parse_file(srt_path)
    interpolator = TelemetryInterpolator(records)
    print(f"[Stage 1] Loaded {len(records)} telemetry records.")

    # 2. Keyframe Extraction
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = 640.0 / max(orig_w, orig_h)
    img_w = int(orig_w * scale)
    img_h = int(orig_h * scale)

    # Focal length for street facade depth geometry
    fx = 0.85 * img_w
    fy = fx
    cx = img_w / 2.0
    cy = img_h / 2.0

    step_interval = max(1, total_frames // max_keyframes)
    keyframes = []
    frame_idx = 0

    print(f"[Stage 2] Sampling {max_keyframes} keyframes (step: {step_interval})...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step_interval == 0:
            resized = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
            pts_ms = int((frame_idx / fps) * 1000)
            lat, lon, alt, pitch = interpolator.query_pose(pts_ms)
            keyframes.append((resized, alt, pitch))

        frame_idx += 1
        if len(keyframes) >= max_keyframes:
            break
    cap.release()
    print(f"[Stage 2] Extracted {len(keyframes)} keyframes.")

    # 3. Model Engine
    depth_engine = DepthEstimatorPipeline(device=device)

    # 4. Dense Reprojection
    u, v = np.meshgrid(np.arange(img_w), np.arange(img_h))
    accumulated_pcd = o3d.geometry.PointCloud()
    prev_down = None
    T_accum = np.eye(4)
    forward_stride = 0.50  # 50cm flight advance per keyframe

    print("[Stage 4] Generating clean metric point clouds (sky masked)...")
    for idx, (frame, alt, pitch) in enumerate(keyframes):
        depth_map, valid_mask = depth_engine.estimate_depth(
            frame, img_h, img_w, z_near=3.0, z_far=22.0
        )

        # Exclude extreme image borders
        border = 8
        valid_mask[:border, :] = False
        valid_mask[-border:, :] = False
        valid_mask[:, :border] = False
        valid_mask[:, -border:] = False

        z = depth_map[valid_mask]
        x = (u[valid_mask] - cx) * z / fx
        y = (v[valid_mask] - cy) * z / fy

        pts_cam = np.stack([x, y, z], axis=-1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)[valid_mask] / 255.0

        cur_pcd = o3d.geometry.PointCloud()
        cur_pcd.points = o3d.utility.Vector3dVector(pts_cam[::2])
        cur_pcd.colors = o3d.utility.Vector3dVector(rgb[::2])
        cur_down = cur_pcd.voxel_down_sample(voxel_size=0.08)

        if prev_down is not None:
            # Safe forward initial transform
            T_init = np.eye(4)
            T_init[2, 3] = forward_stride

            cur_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=25))
            prev_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=25))

            reg = o3d.pipelines.registration.registration_icp(
                cur_down,
                prev_down,
                max_correspondence_distance=0.4,
                init=T_init,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
            )

            # Restrict rotation to prevent map twisting
            R_mat = reg.transformation[:3, :3]
            trace_val = np.clip((np.trace(R_mat) - 1.0) / 2.0, -1.0, 1.0)
            rot_deg = math.degrees(math.acos(trace_val))

            if reg.fitness > 0.50 and rot_deg < 10.0:
                T_accum = T_accum @ reg.transformation
            else:
                T_accum = T_accum @ T_init

        world_pcd = cur_down.transform(T_accum)
        accumulated_pcd += world_pcd
        prev_down = cur_down

        if idx % 3 == 0 or idx == len(keyframes) - 1:
            print(f" -> Processed & Aligned keyframe {idx+1}/{len(keyframes)}")

    # 5. Outlier Filtering
    print("[Stage 5] Filtering spatial noise...")
    accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.07)
    clean_pcd, _ = accumulated_pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1.1)

    print(f"[Stage 5] Structured points: {len(clean_pcd.points):,}")
    o3d.io.write_point_cloud(str(output_pcd_path), clean_pcd)
    print(f"[Stage 5] Saved clean point cloud to: {output_pcd_path}")

    # 6. Surface Meshing
    print("[Stage 6] Reconstructing surface mesh...")
    clean_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=30))
    clean_pcd.orient_normals_consistent_tangent_plane(k=20)

    # Ball Pivoting Reconstruction (avoids ballooning)
    nn_dist = np.mean(clean_pcd.compute_nearest_neighbor_distance())
    radii = [nn_dist * 1.5, nn_dist * 3.0, nn_dist * 6.0]
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        clean_pcd, o3d.utility.DoubleVector(radii)
    )

    if len(mesh.triangles) < 500:
        # Fallback to bounded Poisson if BPA is too sparse
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            clean_pcd, depth=8, scale=1.05, linear_fit=True
        )
        bbox = clean_pcd.get_axis_aligned_bounding_box().scale(1.05, clean_pcd.get_center())
        mesh = mesh.crop(bbox)

    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(output_mesh_path), mesh)
    print(f"[Stage 6] Mesh saved to: {output_mesh_path} ({len(mesh.triangles):,} triangles)")

    print("\n" + "=" * 60)
    print(" Pipeline complete: output/model_georeferenced.ply")
    print(" Point cloud saved: output/debug_points.ply")
    print(" Run 'python scripts/visualize_mesh.py' to view.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/zurich_flight.mp4")
    parser.add_argument("--srt", default="data/zurich_flight.srt")
    parser.add_argument("--keyframes", type=int, default=15)
    args = parser.parse_args()
    run_helios_pipeline(args.video, args.srt, max_keyframes=args.keyframes)