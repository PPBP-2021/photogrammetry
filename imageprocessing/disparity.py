from typing import List
from typing import Tuple
from typing import Union

import cv2
import numpy as np


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
