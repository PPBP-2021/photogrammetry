from typing import Callable

import stl
import numpy as np
from stl.mesh import Mesh
import cv2
import image_utils


from imageprocessing.segmentation import segmentate_grayscale


def litophane_from_image(
    seg_img: np.ndarray,
    resolution: float = 1.0,
    z_scale: Callable[[float], float] = lambda z: z
) -> Mesh:
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

    vertices = []
    faces = []

    X = np.array([i/height for i in range(width)]*height)
    Y = np.array([(height-i//width)/height for i in range(height*width)])
    Z = np.array([z_scale(pixel[2]/255) for pixel in seg_img.reshape((-1, 3))])

    vertices = np.column_stack((X, Y, Z))
    vertices = vertices.reshape((height, width, 3))

    for i in range(height-1):
        for j in range(width-1):
            top_left = vertices[i][j]
            top_right = vertices[i][j+1]
            bottom_right = vertices[i+1][j+1]
            bottom_left = vertices[i+1][j]

            if top_left[2] == 0 or bottom_right[2] == 0:
                continue

            if bottom_left[2] != 0:
                faces.append([top_left, bottom_left, bottom_right])
            if top_right[2] != 0:
                faces.append([top_left, top_right, bottom_right])

    faces = np.array(faces)
    mesh = Mesh(np.zeros(faces.shape[0], dtype=Mesh.dtype))
    mesh.vectors = faces

    return mesh


def litophane_from_stereo(
    img_left: np.ndarray,
    img_right: np.ndarray,
    baseline: float,
    focal_length: float,
    fov: float,
    resolution: float = 1.0,
    z_scale: Callable[[float], float] = lambda z: z
) -> Mesh:
    """Create a 3d model from 2 images using stereo depth estimation.

    Parameters
    ----------
    img_left : np.ndarray
        [description]
    img_right : np.ndarray
        [description]
    baseline : float
        [description]
    focal_length : float
        [description]
    fov : float
        [description]
    resolution : float, optional
        [description], by default 1.0
    z_scale : Callable[[float], float], optional
        [description], by default lambdaz:z

    Returns
    -------
    Mesh
        [description]
    """

    pass
