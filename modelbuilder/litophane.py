from typing import Callable
from typing import List
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import open3d

import image_utils


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


def litophane_from_stereo(
    img_left: np.ndarray,
    img_right: np.ndarray,
    baseline: float,
    focal_length: float,
    fov: float,
    resolution: float = 1.0,
    z_scale: Callable[[float], float] = lambda z: z,
    match_features: bool = False,
) -> open3d.geometry.TriangleMesh:
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

    # start with optional resize of the images
    img_left = cv2.resize(img_left, (0, 0), fx=resolution, fy=resolution)
    img_right = cv2.resize(img_right, (0, 0), fx=resolution, fy=resolution)

    # grayscale
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # extract the current image size
    height, width = img_left.shape

    left_points, right_points, _ = match_keypoints(img_left, img_right)

    disparity = calculate_disparity(left_points, right_points, img_left, img_right)

    image_utils.show_img_grayscale(disparity, "Disparity map")

    return
    """
    X = np.array([i/height for i in range(width)]*height)
    Y = np.array([(height-i//width)/height for i in range(height*width)])

    # calculate the focal length in pixels
    focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi/180)
    focal_length = 0.8 * width
    Z = z_scale(((baseline * focal_length) / disparity).reshape(height*width))

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
    """


def match_keypoints(
    img_left: np.ndarray, img_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = img_left.shape

    # get the important keypoints and descriptors for both images
    sift = cv2.SIFT_create()
    kp_left: Tuple[cv2.KeyPoint]
    des_left: np.ndarray
    kp_left, des_left = sift.detectAndCompute(img_left, None)

    kp_right: Tuple[cv2.KeyPoint]
    des_right: np.ndarray
    kp_right, des_right = sift.detectAndCompute(img_right, None)

    # Match the descriptors between both images
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)

    matches: Union[Tuple[cv2.DMatch], List[cv2.DMatch]]
    matches = matcher.knnMatch(des_left, des_right, k=2)

    # Only use the best matches
    # based on https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
    # Lowes Paper: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matches_mask = [[0, 0] for i in range(len(matches))]
    pts1 = []
    pts2 = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            # Keep this keypoint pair
            matches_mask[i] = [1, 0]
            pts2.append(kp_right[m.trainIdx].pt)
            pts1.append(kp_left[m.queryIdx].pt)

    draw_params = dict(
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_DEFAULT,
    )

    # Show the best matches between both images
    img_with_matches = cv2.drawMatchesKnn(
        img_left, kp_left, img_right, kp_right, matches, None, **draw_params
    )

    return np.array(pts1, dtype=int), np.array(pts2, dtype=int), img_with_matches


def find_fundamental_matrix(
    left_points: np.ndarray, right_points: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    return cv2.findFundamentalMat(
        left_points, right_points, cv2.FM_RANSAC
    )  # type: ignore


def calculate_disparity(
    left_points: np.ndarray,
    right_points: np.ndarray,
    img_left: np.ndarray,
    img_right: np.ndarray,
    minDisparity=-1,
    numDisparities=5 * 16,
    window_size=5,
    disp12MaxDiff=12,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32,
) -> np.ndarray:
    height, width = img_left.shape

    # We need to find our Fundamental Matrix
    fundamental: np.ndarray
    inliers: np.ndarray
    fundamental, inliers = find_fundamental_matrix(left_points, right_points)

    # keep only the
    left_points = left_points[inliers.ravel() == 1]
    right_points = right_points[inliers.ravel() == 1]

    # get the homography matrices for each image
    h_left: np.ndarray
    h_right: np.ndarray
    _, h_left, h_right = cv2.stereoRectifyUncalibrated(
        left_points, right_points, fundamental, imgSize=(width, height)
    )

    left_rectified = cv2.warpPerspective(img_left, h_left, (width, height))
    right_rectified = cv2.warpPerspective(img_right, h_right, (width, height))

    # disparity range has to be tuned for each image pair
    stereo = cv2.StereoSGBM_create(
        minDisparity=minDisparity,
        numDisparities=numDisparities,
        blockSize=window_size,
        P1=8 * 1 * window_size ** 2,
        P2=32 * 1 * window_size ** 2,
        disp12MaxDiff=disp12MaxDiff,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    disparity = stereo.compute(left_rectified, right_rectified)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity = cv2.normalize(
        disparity, disparity, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX
    )
    disparity = np.uint8(disparity)

    return disparity
