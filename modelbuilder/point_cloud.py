from typing import Callable
from typing import cast

import numpy as np
import open3d


def calculate_stereo_point_cloud(
    disparity: np.ndarray,
    baseline: float,
    fov: float,
    z_scale: Callable[[float], float] = lambda z: z,
) -> open3d.geometry.PointCloud:
    height, width = disparity.shape

    X = np.array([i / height for i in range(width)] * height)
    Y = np.array([(height - i // width) / height for i in range(height * width)])

    # calculate the focal length in pixels
    focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi / 180)

    disparity = disparity.astype(float)
    disparity[disparity == 0] = np.inf

    Z: np.ndarray = cast(
        np.ndarray, z_scale((baseline * focal_length) / disparity)
    ).reshape(height * width)

    xyz = np.stack((X, Y, Z), axis=1)

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(xyz)

    return point_cloud


def calculate_point_cloud_final_model(
    disparity_front: np.ndarray,
    disparity_back: np.ndarray,
    disparity_left: np.ndarray,
    disparity_right: np.ndarray,
    baseline: float,
    fov: float,
    z_scale: Callable[[float], float] = lambda z: z,
) -> open3d.geometry.PointCloud:
    height, width = disparity_front.shape
    # calculate the focal length in pixels
    focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi / 180)

    disparity_front = disparity_front.astype(float)
    disparity_front[disparity_front == 0] = np.inf
    disparity_back = disparity_back.astype(float)
    disparity_back[disparity_back == 0] = np.inf
    disparity_left = disparity_left.astype(float)
    disparity_left[disparity_left == 0] = np.inf
    disparity_right = disparity_right.astype(float)
    disparity_right[disparity_right == 0] = np.inf

    X = np.array([i / height for i in range(width)] * height)
    Y = np.array([(height - i // width) / height for i in range(height * width)])

    z_scale = cast(Callable[[np.ndarray], np.ndarray], z_scale)
    Z_front = z_scale((baseline * focal_length) / disparity_front).reshape(
        height * width
    )
    Z_back = z_scale((baseline * focal_length) / disparity_back).reshape(height * width)
    Z_left = z_scale((baseline * focal_length) / disparity_left).reshape(height * width)
    Z_right = z_scale((baseline * focal_length) / disparity_right).reshape(
        height * width
    )

    # Remove points with zero depth, these are not useful for the final model
    def cutoff(Z, X, Y):
        keep = np.where(Z > 0)[0]
        X = X[keep]
        Y = Y[keep]
        Z = Z[keep]
        return X, Y, Z

    f_x, f_y, f_z = cutoff(Z_front, X, Y)
    b_x, b_y, b_z = cutoff(Z_back, X, Y)
    l_x, l_y, l_z = cutoff(Z_left, X, Y)
    r_x, r_y, r_z = cutoff(Z_right, X, Y)

    # width/height is the max X value
    ratio = width / height
    # Use -Z for front to fix X axis
    # TODO fix rotations and transformations
    xyz_front = np.stack((f_x - ratio / 2, f_y, f_z - ratio), axis=1)
    xyz_left = np.stack((l_z - ratio, l_y, l_x - ratio / 2), axis=1)
    xyz_back = np.stack((b_x - ratio / 2, b_y, (-b_z) + ratio), axis=1)
    xyz_right = np.stack((-r_z + ratio, r_y, r_x - ratio / 2), axis=1)

    total_point_cloud = np.concatenate(
        (xyz_front, xyz_left, xyz_back, xyz_right), axis=0
    )
    total_point_cloud = np.concatenate((total_point_cloud, np.zeros((1, 3))), axis=0)

    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(total_point_cloud)
    return point_cloud
