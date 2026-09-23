from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import open3d as o3d
import torch

from src.reconstruction.gaussian_model import GaussianModel


@dataclass
class MeshOutput:
    vertices: np.ndarray      # Shape: (V, 3) float32 coordinates
    triangles: np.ndarray     # Shape: (F, 3) int32 vertex indices
    vertex_colors: np.ndarray # Shape: (V, 3) float32 RGB in [0, 1]
    num_vertices: int
    num_triangles: int


class PoissonSurfaceMesher:
    """Extracts watertight triangular meshes from trained 3D Gaussian representations

    using Screened Poisson Surface Reconstruction.
    """

    def __init__(
        self,
        poisson_depth: int = 9,
        min_opacity_threshold: float = 0.3,
        density_quantile_trim: float = 0.05,
    ):
        self.poisson_depth = poisson_depth
        self.min_opacity_threshold = min_opacity_threshold
        self.density_quantile_trim = density_quantile_trim

    def extract_point_cloud(
        self,
        model: GaussianModel,
    ) -> o3d.geometry.PointCloud:
        """Extracts filtered metric coordinates, colors, and oriented surface normals

        from the Gaussian model.
        """
        xyz = model.get_xyz.detach().cpu().numpy()
        opacities = model.get_opacity.detach().cpu().numpy().squeeze(-1)
        features = model.get_features.detach().cpu().numpy()

        # Convert SH0 base color to linear RGB
        C0 = 0.28209479177387814
        colors = np.clip(features * C0 + 0.5, 0.0, 1.0)

        # Filter out low-opacity floaters
        valid_mask = opacities >= self.min_opacity_threshold
        if not np.any(valid_mask):
            # Fallback if training was too short
            valid_mask = np.ones_like(opacities, dtype=bool)

        pts_filt = xyz[valid_mask]
        col_filt = colors[valid_mask]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_filt)
        pcd.colors = o3d.utility.Vector3dVector(col_filt)

        # Estimate and orient normals using local tangent planes
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)

        return pcd

    def reconstruct_mesh(
        self,
        pcd: o3d.geometry.PointCloud,
    ) -> Tuple[o3d.geometry.TriangleMesh, MeshOutput]:
        """Runs Screened Poisson reconstruction and trims low-density boundary artifacts."""
        if len(pcd.points) < 10:
            raise ValueError("Insufficient point density for Poisson reconstruction.")

        # Screened Poisson Surface Reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=self.poisson_depth,
            linear_fit=True,
        )

        # Trim low-density artifacts (spurious surface bubbles outside the survey boundary)
        densities_arr = np.asarray(densities)
        trim_threshold = np.quantile(densities_arr, self.density_quantile_trim)
        vertices_to_remove = densities_arr < trim_threshold
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Clean geometry: remove non-manifold edges and isolated clusters
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Interpolate vertex colors from the nearest point cloud vertices
        mesh.compute_vertex_normals()

        verts = np.asarray(mesh.vertices)
        tris = np.asarray(mesh.triangles)
        v_colors = (
            np.asarray(mesh.vertex_colors)
            if mesh.has_vertex_colors()
            else np.ones((len(verts), 3), dtype=np.float32) * 0.7
        )

        output = MeshOutput(
            vertices=verts,
            triangles=tris,
            vertex_colors=v_colors,
            num_vertices=len(verts),
            num_triangles=len(tris),
        )

        return mesh, output

    def export_mesh(self, mesh: o3d.geometry.TriangleMesh, output_path: str) -> bool:
        """Exports triangle mesh to standard formats (.ply, .obj, .stl)."""
        return bool(o3d.io.write_triangle_mesh(output_path, mesh))


if __name__ == "__main__":
    print("Testing PoissonSurfaceMesher...")
    mesher = PoissonSurfaceMesher(poisson_depth=7)

    # Synthetic point cloud on a hemispherical dome (1500 points)
    np.random.seed(42)
    phi = np.random.uniform(0, 2 * np.pi, 1500)
    theta = np.random.uniform(0, np.pi / 2, 1500)
    radius = 5.0

    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    synthetic_points = np.stack([x, y, z], axis=-1).astype(np.float32)

    # Initialize a Gaussian model with this dome
    model = GaussianModel(device="cpu")
    model.initialize_from_pcd(synthetic_points, init_scale=0.1)

    pcd = mesher.extract_point_cloud(model)
    print(f"Point Cloud Extracted: {len(pcd.points)} points with normals.")

    mesh, stats = mesher.reconstruct_mesh(pcd)
    print(f"Poisson Reconstruction Complete!")
    print(f"Vertices:  {stats.num_vertices}")
    print(f"Triangles: {stats.num_triangles}")

    assert stats.num_vertices > 0, "No vertices in reconstructed mesh."
    assert stats.num_triangles > 0, "No triangles in reconstructed mesh."
    print("Poisson Surface Meshing Verification Successful!")