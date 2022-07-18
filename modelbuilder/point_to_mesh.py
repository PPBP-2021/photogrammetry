import numpy as np
import open3d as o3d


def point_cloud_to_mesh(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
    """Takes a Point Cloud, estimates the new normals and returns a TriangleMesh."""

    pcd.normals = o3d.utility.Vector3dVector(
        np.zeros((1, 3))
    )  # invalidate existing normals

    pcd.estimate_normals()
    radii = [0.005, 0.01, 0.02, 0.04]
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return rec_mesh
