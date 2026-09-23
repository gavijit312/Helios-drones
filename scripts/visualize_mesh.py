import open3d as o3d


def view():
    path = "output/model_georeferenced.ply"
    print(f"Loading mesh: {path}")
    mesh = o3d.io.read_triangle_mesh(path)
    if len(mesh.vertices) == 0:
        print("Mesh not found or empty. Run src/pipeline.py first.")
        return

    mesh.compute_vertex_normals()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0)
    o3d.visualization.draw_geometries([mesh, coord], window_name="Helios 3D Recon Mesh")


if __name__ == "__main__":
    view()