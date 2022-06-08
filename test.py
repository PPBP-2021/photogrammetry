import json
import math

import cv2
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

import image_utils as imgutils
import modelbuilder
from imageprocessing import disparity as dp

"""SIMPLE DISPARITY MAP ONLY USING CV2.STEREO_BM"""
stereo_left_img = cv2.imread("dashboard/assets/Manuel_L.jpg", 0)
stereo_right_img = cv2.imread("dashboard/assets/Manuel_R.jpg", 0)
width, height = stereo_left_img.shape
with open("dashboard/assets/room.json") as fp:
    camera_parameters = json.load(fp)
    baseline = camera_parameters["baseline"]
    fov = camera_parameters["fov"]
'''stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(stereo_left_img, stereo_right_img)
imgutils.show_img_grayscale(disparity)

# calculate the stereo_point_cloud
"""lito_point_cloud = modelbuilder.point_cloud.calculate_stereo_point_cloud(
    disparity, baseline, fov
)

imgutils.show_point_cloud(lito_point_cloud)"""

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
imgutils.show_img_grayscale(disparity_SGBM)'''

"""MORE COMPLICATED DISPARITY MAP USING KEYPOINT MATCHING AND FUNDAMENTAL MATRIX"""
# keypoint matching
left_points, right_points, _ = dp.match_keypoints(stereo_left_img, stereo_right_img)

fundamental: np.ndarray
inliers: np.ndarray
# getting the fundamental matrix
fundamental, inliers = dp.find_fundamental_matrix(left_points, right_points)

left_points = left_points[inliers.ravel() == 1]
right_points = right_points[inliers.ravel() == 1]

# get the homography matrices for each image
h_left: np.ndarray
h_right: np.ndarray
_, h_left, h_right = cv2.stereoRectifyUncalibrated(
    np.float32(left_points),
    np.float32(right_points),
    fundamental,
    imgSize=(width, height),
)

left_rectified = cv2.warpPerspective(stereo_left_img, h_left, (width, height))
right_rectified = cv2.warpPerspective(stereo_right_img, h_right, (width, height))

fig = make_subplots(rows=1, cols=2)
fig.add_trace(px.imshow(left_rectified).data[0], row=1, col=1)
fig.add_trace(px.imshow(right_rectified).data[0], row=1, col=2)
fig.show()

stereo = cv2.StereoSGBM_create(
    minDisparity=-128,
    numDisparities=192,  # max_disp has to be dividable by 16 f. E. HH 192, 256
    blockSize=11,
    P1=8 * 1 * 11 * 11,
    # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    P2=32 * 1 * 11 * 11,
    disp12MaxDiff=0,
    uniquenessRatio=5,
    speckleWindowSize=200,
    speckleRange=2,
)
disparity_SGBM = stereo.compute(left_rectified, right_rectified)

# Normalize the values to a range from 0..255 for a grayscale image
disparity_SGBM = cv2.normalize(
    disparity_SGBM, disparity_SGBM, alpha=255, beta=0, norm_type=cv2.NORM_MINMAX
)
disparity_SGBM = np.uint8(disparity_SGBM)
disparity_SGBM = 255 - disparity_SGBM

imgutils.show_img_grayscale(disparity_SGBM)
