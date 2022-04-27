import json
import math
from typing import Tuple
from typing import Union

import cv2
import numpy as np
import pandas as pd
import plotly.express as px

import image_utils as imgutils
import imageprocessing as imgp
import modelbuilder

"""SIMPLE DISPARITY MAP ONLY USING CV2.STEREO_BM"""
stereo_left_img = cv2.imread("dashboard/assets/room_L.png", 0)
stereo_right_img = cv2.imread("dashboard/assets/room_R.png", 0)
width, height = stereo_left_img.shape
with open("dashboard/assets/room.json") as fp:
    camera_parameters = json.load(fp)
    baseline = camera_parameters["baseline"]
    fov = camera_parameters["fov"]
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(stereo_left_img, stereo_right_img)
imgutils.show_img_grayscale(disparity)

# calculate the stereo_litophane
lito_point_cloud = modelbuilder.litophane.calculate_stereo_litophane_point_cloud(
    disparity, baseline, fov
)

imgutils.show_point_cloud(lito_point_cloud)

"""USING STEREO_SGBM FOR THE DISPARITY MAP"""
stereo = cv2.StereoSGBM_create(
    minDisparity=-48,
    numDisparities=64,
    blockSize=5,
    uniquenessRatio=5,
    speckleWindowSize=200,
    speckleRange=5,
    disp12MaxDiff=5,
    P1=8 * 1 * 5 * 5,
    P2=32 * 1 * 5 * 5,
    mode=cv2.StereoSGBM_MODE_SGBM_3WAY,
)

disparity_SGBM = stereo.compute(stereo_left_img, stereo_right_img)
"""disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,beta=0,cv2.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)"""
imgutils.show_img_grayscale(disparity_SGBM)

"""MORE COMPLICATED DISPARITY MAP USING KEYPOINT MATCHING AND FUNDAMENTAL MATRIX"""
# keypoint matching
left_points, right_points, _ = modelbuilder.match_keypoints(
    stereo_left_img, stereo_right_img
)

fundamental: np.ndarray
inliers: np.ndarray
# getting the fundamental matrix
fundamental, inliers = modelbuilder.find_fundamental_matrix(left_points, right_points)

left_points = left_points[inliers.ravel() == 1]
right_points = right_points[inliers.ravel() == 1]

# get the homography matrices for each image
h_left: np.ndarray
h_right: np.ndarray
_, h_left, h_right = cv2.stereoRectifyUncalibrated(
    left_points, right_points, fundamental, imgSize=(width, height)
)

left_rectified = cv2.warpPerspective(stereo_left_img, h_left, (width, height))
right_rectified = cv2.warpPerspective(stereo_right_img, h_right, (width, height))

# SGBM Parameters -----------------
window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely

left_matcher = cv2.StereoSGBM_create(
    minDisparity=-1,
    numDisparities=15 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=window_size,
    P1=8 * 3 * window_size,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
# FILTER Parameters
lmbda = 80000
sigma = 1.3
visual_multiplier = 6

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)

wls_filter.setSigmaColor(sigma)
displ = left_matcher.compute(left_rectified, right_rectified)  # .astype(np.float32)/16
dispr = right_matcher.compute(right_rectified, left_rectified)  # .astype(np.float32)/16
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(
    displ, left_rectified, None, dispr
)  # important to put "imgL" here!!!

filteredImg = cv2.normalize(
    src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX
)
filteredImg = np.uint8(filteredImg)

imgutils.show_img_grayscale(filteredImg)
