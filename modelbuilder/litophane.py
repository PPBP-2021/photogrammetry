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
    z_scale: Callable[[float], float] = lambda z: z,
    match_features: bool = False
) -> Mesh:
    """Create a 3d model from 2 images using stereo depth estimation.

    Parameters
    ----------
    img_left : np.ndarray
        The Left image data in BGR format.
    img_right : np.ndarray
        The Right image data in BGR format.
    baseline : float
        Distance between both cameras in mm.
    focal_length : float
        Focal Lenght of the camera in mm.
    fov : float
        Horizontal FOV of the camera.
    resolution : float, optional
        Changes the images resolution before calculating the depth, increses performance if lower than 1.0, by default 1.0
    z_scale : Callable[[float], float], optional
        Scale the calculated depth value using this function, by default lambdaz:z
    match_features : bool, optional
        Uses opencvs feature matching algorithm if true, else calculate disparity by hand assuming ideal image alignment.


    Returns
    -------
    Mesh
        The Mesh using the calculated depth information from both images.
    """
    # img_left = cv2.resize(img_left, (0, 0), fx=resolution, fy=resolution)
    # img_right = cv2.resize(img_right, (0, 0), fx=resolution, fy=resolution)
    # img_left = cv2.medianBlur(img_left, 5)
    # img_right = cv2.medianBlur(img_right, 5)

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    height, width = img_left.shape

    if match_features:
        win_size = 3
        min_disp = 50
        num_disp = 16 * 30
        stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                       numDisparities=num_disp,
                                       blockSize=5,
                                       uniquenessRatio=0,
                                       speckleWindowSize=5,
                                       speckleRange=120,
                                       disp12MaxDiff=20,
                                       P1=8*3*win_size**2,
                                       P2=32*3*win_size**2)

        stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=25)
        disparity = stereo.compute(img_left, img_right).astype(np.float32)

    else:
        ...
        # normalize both images to the same rect size
        # calculate disparity

    image_utils.show_img_grayscale(disparity)

    raise Exception("break lul")

    disparity[disparity == 0] = -1

    X = np.array([i/height for i in range(width)]*height)
    Y = np.array([(height-i//width)/height for i in range(height*width)])

    Z = z_scale(((baseline * focal_length) / disparity).reshape(height*width))

    Z[Z < 0] = 0
    Z[Z > 10] = 10

    vertices = []
    faces = []

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
