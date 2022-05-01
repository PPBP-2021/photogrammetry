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
