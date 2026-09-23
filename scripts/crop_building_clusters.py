from pathlib import Path
import numpy as np
import open3d as o3d


def extract_building_crops(
    ply_path: str = "data/SensatUrban/original_block_ply/birmingham_block_0.ply",
    output_dir: str = "data/processed/building_crops",
    num_crops: int = 200,
    target_points: int = 2048,
    tile_size: float = 35.0,
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Occlusion Pipeline] Reading: {ply_path}")
    pcd = o3d.io.read_point_cloud(ply_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.25)

    pts = np.asarray(pcd.points)
    print(f"Total points after downsampling: {len(pts):,}")

    min_b = pts.min(axis=0)
    max_b = pts.max(axis=0)

    # Divide block into spatial tiles across X and Y
    x_bins = np.arange(min_b[0], max_b[0], tile_size)
    y_bins = np.arange(min_b[1], max_b[1], tile_size)

    saved = 0
    print(f"Slicing spatial tiles ({tile_size}m x {tile_size}m)...")

    for x in x_bins:
        for y in y_bins:
            if saved >= num_crops:
                break

            mask = (
                (pts[:, 0] >= x) & (pts[:, 0] < x + tile_size) &
                (pts[:, 1] >= y) & (pts[:, 1] < y + tile_size)
            )
            tile_pts = pts[mask]

            # Keep tiles with sufficient structural density
            if len(tile_pts) < target_points:
                continue

            # Filter out flat ground: retain only tiles with structural height variance (> 6m)
            height_range = tile_pts[:, 2].max() - tile_pts[:, 2].min()
            if height_range < 6.0:
                continue

            # Sample fixed point density
            choice = np.random.choice(len(tile_pts), target_points, replace=False)
            sampled = tile_pts[choice].copy()

            # Canonical zero-centering and scale normalization to [-1, 1]
            centroid = np.mean(sampled, axis=0)
            sampled -= centroid
            scale = np.max(np.sqrt(np.sum(sampled**2, axis=1))) + 1e-6
            norm_pts = (sampled / scale).astype(np.float32)

            out_file = out_dir / f"building_{saved:04d}.npy"
            np.save(str(out_file), norm_pts)
            saved += 1

    print(f"Successfully generated {saved} structural building models in {output_dir}")


if __name__ == "__main__":
    extract_building_crops()