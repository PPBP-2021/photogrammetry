from typing import Callable
from typing import cast

import cv2
import numpy as np
import open3d


def litophane_from_image(
    seg_img: np.ndarray,
    resolution: float = 1.0,
    z_scale: Callable[[float], float] = lambda z: z,
) -> open3d.geometry.TriangleMesh:
    """Construct a 3D litophane of a segmentated image.

    Parameters
    ----------
    seg_img : np.ndarray
        Image segmentated into foreground, background. Background is completly black.

    resolution : float
        Value between 0 and 1; the percentage of pixels used.

    z_scale : Callable[[float], float] to apply on the depth value. Original Z between 0 and 1.
    """

    seg_img = cv2.resize(seg_img, (0, 0), fx=resolution, fy=resolution)

    seg_img = cv2.medianBlur(seg_img, 5)

    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2HSV)

    height, width, _ = seg_img.shape

    mesh = open3d.geometry.TriangleMesh()

    X = np.array([i / height for i in range(width)] * height)
    Y = np.array([(height - i // width) / height for i in range(height * width)])
    Z = np.array([z_scale(pixel[2] / 255) for pixel in seg_img.reshape((-1, 3))])

    vertices = np.column_stack((X, Y, Z))
    mesh.vertices = open3d.utility.Vector3dVector(vertices)

    tris = []
    for i in range(height - 1):
        for j in range(width - 1):

            index = i * width + j

            tris.append([index, index + 1, index + width])

            tris.append([index + width, index + 1, index + width + 1])

    tris = np.array(tris).astype(int).reshape((-1, 3))
    mesh.triangles = open3d.utility.Vector3iVector(tris)

    return mesh


def calculate_stereo_litophane_mesh(
    disparity: np.ndarray,
    baseline: float,
    fov: float,
    z_scale: Callable[[float], float] = lambda z: z,
) -> open3d.geometry.TriangleMesh:
    # extract the current image size
    height, width = disparity.shape

    mesh = open3d.geometry.TriangleMesh()

    X = np.array([i / height for i in range(width)] * height)
    Y = np.array([(height - i // width) / height for i in range(height * width)])

    # calculate the focal length in pixels
    focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi / 180)

    disparity = disparity.astype(float)
    disparity[disparity == 0] = np.inf

    Z = z_scale(((baseline * focal_length) / disparity).reshape(height * width))
    Z = -Z  # invert the z axis

    vertices = np.column_stack((X, Y, Z))
    mesh.vertices = open3d.utility.Vector3dVector(vertices)

    tris = []
    for i in range(height - 1):
        for j in range(width - 1):

            index = i * width + j

            tris.append([index, index + 1, index + width])

            tris.append([index + width, index + 1, index + width + 1])

    tris = np.array(tris).astype(int).reshape((-1, 3))
    mesh.triangles = open3d.utility.Vector3iVector(tris)

    return mesh


def calculate_stereo_litophane_point_cloud(
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
