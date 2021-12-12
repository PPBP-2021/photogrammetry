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
    seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)

    height, width = seg_img.shape

    vertices = []
    faces = []

    for i, row in enumerate(seg_img):
        for j, pixel in enumerate(row):
            Z = pixel / 255
            vertices.append([j/height, (height-i)/height, z_scale(Z)])

    vertices = np.array(vertices)
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
