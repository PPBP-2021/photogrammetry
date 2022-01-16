from typing import Callable
from numpy.lib.function_base import disp

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
    sensor width : float

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
    img_left = cv2.resize(img_left, (0, 0), fx=resolution, fy=resolution)
    img_right = cv2.resize(img_right, (0, 0), fx=resolution, fy=resolution)

    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    height, width = img_left.shape

    orb = cv2.ORB_create(int(np.sqrt(height*width)))
    kp_left, des_left = orb.detectAndCompute(img_left, None)
    kp_right, des_right = orb.detectAndCompute(img_right, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_left, des_right)

    # take the best matches
    matches = sorted(matches, key=lambda x: x.distance)[:30]

    img_with_matches = cv2.drawMatches(
        img_left, kp_left, img_right, kp_right, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    image_utils.show_img(
        img_with_matches, "Found matches between left and right image")
    left_points = []
    right_points = []
    for m in matches:
        left_points.append(kp_left[m.queryIdx].pt)
        right_points.append(kp_right[m.trainIdx].pt)

    left_points = np.array(left_points)
    right_points = np.array(right_points)
    fundamental, mask = cv2.findFundamentalMat(
        left_points, right_points, cv2.FM_RANSAC, 1.0, 0.98)
    _, h_left, h_right = cv2.stereoRectifyUncalibrated(
        left_points, right_points, fundamental, (width, height))

    left_rectified = cv2.warpPerspective(img_left, h_left, (width, height))
    right_rectified = cv2.warpPerspective(img_right, h_right, (width, height))

    rectified_together = np.concatenate(
        (left_rectified, right_rectified), axis=1)
    image_utils.show_img(rectified_together, "Rectified images")

    focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi/180)

    win_size = 3
    min_disp = 50
    num_disp = 16 * 30
    stereo = cv2.StereoSGBM_create(128,31)

    #stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=25)
    disparity = stereo.compute(
        left_rectified, right_rectified).astype(np.float32)

    disparity = np.abs(disparity)
    image_utils.show_img_grayscale(disparity, "Disparity map")

    disparity[disparity == 0] = -1

    X = np.array([i/height for i in range(width)]*height)
    Y = np.array([(height-i//width)/height for i in range(height*width)])
    
    Z = z_scale(((baseline * focal_length) / disparity).reshape(height*width))
    Z[Z<0] = 0

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
