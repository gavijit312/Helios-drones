from pathlib import Path
import numpy as np
import open3d as o3d


class PoissonReconstructor:
    def __init__(self, depth: int = 8, density_quantile: float = 0.06, voxel_size: float = 0.15):
        self.depth = depth
        self.density_quantile = density_quantile
        self.voxel_size = voxel_size

    def reconstruct_surface(
        self,
        point_cloud: o3d.geometry.PointCloud,
        output_ply_path: str = "output/model_georeferenced.ply",
    ) -> o3d.geometry.TriangleMesh:
        Path(output_ply_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"[Meshing] Raw input points: {len(point_cloud.points):,}")
        
        # 1. Downsample to preserve fine building edges at street scale
        pcd = point_cloud.voxel_down_sample(voxel_size=self.voxel_size)
        print(f"[Meshing] Downsampled to {len(pcd.points):,} points (voxel_size={self.voxel_size}m).")

        if len(pcd.points) < 100:
            print("[Meshing] Error: Not enough points to reconstruct mesh.")
            return o3d.geometry.TriangleMesh()

        # 2. Hybrid KDTree Normal Estimation
        print("[Meshing] Estimating surface normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30)
        )

        # 3. Orient normals upwards/towards camera (+Z / +Y)
        pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 0.0, 1.0]))

        # 4. Screened Poisson Surface Reconstruction
        print(f"[Meshing] Executing Screened Poisson Reconstruction (octree depth={self.depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, linear_fit=True
        )

        # 5. Trim low-density surface boundaries
        densities = np.asarray(densities)
        if len(densities) > 0:
            density_thresh = np.quantile(densities, self.density_quantile)
            vertices_to_remove = densities < density_thresh
            mesh.remove_vertices_by_mask(vertices_to_remove)

        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_ply_path, mesh)
        print(f"[Meshing] Saved reconstructed 3D surface to: {output_ply_path}")
        return mesh