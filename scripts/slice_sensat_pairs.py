import math
from pathlib import Path
import cv2
import numpy as np
import open3d as o3d


def slice_sensat_urban(
    ply_path: str = "data/SensatUrban/original_block_ply/birmingham_block_0.ply",
    output_dir: str = "data/processed/depth_pairs",
    num_views: int = 400,
    img_size: int = 518,
    voxel_size: float = 0.2,
):
    out_dir = Path(output_dir)
    rgb_dir = out_dir / "rgb"
    depth_dir = out_dir / "depth_gt"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"Original point count: {len(pcd.points):,}")

    if voxel_size > 0:
        print(f"Voxel downsampling at {voxel_size}m for fast camera projection...")
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f"Downsampled point count: {len(pcd.points):,}")

    pts = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)

    # Center cloud around origin
    min_b = pts.min(axis=0)
    max_b = pts.max(axis=0)
    center = (min_b + max_b) / 2.0
    pts = pts - center

    # Camera intrinsics (84 deg horizontal FOV typical of drone sensors)
    fov_rad = math.radians(84.0)
    f = (img_size / 2.0) / math.tan(fov_rad / 2.0)
    cx, cy = img_size / 2.0, img_size / 2.0

    print(f"Synthesizing {num_views} aerial drone viewpoints...")

    # Grid search across the bounding box
    grid_dim = int(math.ceil(math.sqrt(num_views)))
    x_steps = np.linspace(pts[:, 0].min() * 0.75, pts[:, 0].max() * 0.75, grid_dim)
    y_steps = np.linspace(pts[:, 1].min() * 0.75, pts[:, 1].max() * 0.75, grid_dim)

    max_z = pts[:, 2].max()
    saved = 0

    for x in x_steps:
        for y in y_steps:
            if saved >= num_views:
                break

            # Drone hovering altitude: 35m to 55m above rooftops
            flight_alt = max_z + np.random.uniform(35.0, 55.0)
            cam_pos = np.array([x, y, flight_alt], dtype=np.float32)

            # Camera pitch tilt (15 deg to 35 deg oblique downward) & random yaw heading
            pitch = math.radians(np.random.uniform(15.0, 35.0))
            yaw = np.random.uniform(0.0, 2.0 * math.pi)

            # Rotation matrix: Yaw then Pitch
            c_y, s_y = math.cos(yaw), math.sin(yaw)
            c_p, s_p = math.cos(pitch), math.sin(pitch)

            R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]], dtype=np.float32)
            R_x = np.array([[1, 0, 0], [0, c_p, -s_p], [0, s_p, c_p]], dtype=np.float32)
            R = R_x @ R_z

            # Transform world points into camera space
            rel_pts = pts - cam_pos
            cam_pts = rel_pts @ R.T

            # Camera looks along negative Z in standard graphics, or positive Z in CV
            # Depth Z is distance along optical axis
            z = -cam_pts[:, 2]

            valid = (z > 3.0) & (z < 120.0)
            if np.sum(valid) < 5000:
                continue

            z_val = z[valid]
            cam_valid = cam_pts[valid]
            cols_valid = colors[valid]

            # Perspective projection
            u = (cam_valid[:, 0] * f / z_val + cx).astype(np.int32)
            v = (cam_valid[:, 1] * f / z_val + cy).astype(np.int32)

            in_bounds = (u >= 0) & (u < img_size) & (v >= 0) & (v < img_size)
            if np.sum(in_bounds) < 3500:
                continue

            u_f = u[in_bounds]
            v_f = v[in_bounds]
            z_f = z_val[in_bounds]
            c_f = cols_valid[in_bounds]

            # Rasterize with depth buffering
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            depth_map = np.zeros((img_size, img_size), dtype=np.float32)

            # Sort farthest to closest
            sort_order = np.argsort(-z_f)
            u_s = u_f[sort_order]
            v_s = v_f[sort_order]
            z_s = z_f[sort_order]
            c_s = c_f[sort_order]

            img[v_s, u_s] = c_s
            depth_map[v_s, u_s] = z_s

            # Infill small sparse gaps from voxel sampling
            hole_mask = (depth_map == 0).astype(np.uint8)
            depth_infilled = cv2.inpaint(
                depth_map, hole_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA
            )

            # Save frame pair
            tag = f"sensat_{saved:04d}"
            cv2.imwrite(str(rgb_dir / f"{tag}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            np.save(str(depth_dir / f"{tag}.npy"), depth_infilled.astype(np.float32))

            saved += 1
            if saved % 50 == 0 or saved == num_views:
                print(f"Generated [{saved}/{num_views}] RGB-D pairs...")

    print(f"Completed! Total {saved} pairs saved to: {output_dir}")


if __name__ == "__main__":
    slice_sensat_urban()