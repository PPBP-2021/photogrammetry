import json

import cv2
import numpy as np
import open3d
import plotly.express as px
from plotly.subplots import make_subplots

import image_utils as imgutils
import modelbuilder as mb
from imageprocessing import disparity as dp
from imageprocessing import rectify

# ToDo: Maybe turn this into a single script with lots of explanation for presentation purposes

"""SIMPLE DISPARITY MAP ONLY USING CV2.STEREO_BM"""
'''stereo_left_img_bgr = cv2.imread("dashboard/assets/Manuel_L.jpg")
stereo_right_img_bgr = cv2.imread("dashboard/assets/Manuel_R.jpg")
stereo_left_img = cv2.cvtColor(stereo_left_img_bgr, cv2.COLOR_BGR2GRAY)
stereo_right_img = cv2.cvtColor(stereo_right_img_bgr, cv2.COLOR_BGR2GRAY)

height, width = stereo_left_img.shape
with open("dashboard/assets/room.json") as fp:
    camera_parameters = json.load(fp)
    baseline = camera_parameters["baseline"]
    fov = camera_parameters["fov"]
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
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
imgutils.show_img_grayscale(disparity_SGBM)

"""MORE COMPLICATED DISPARITY MAP USING KEYPOINT MATCHING AND FUNDAMENTAL MATRIX TO RECTIFY IMAGE"""

rectified_left, rectified_right = rectify.rectify(stereo_left_img, stereo_right_img)

disparity_SGBM = dp.disparity_simple(rectified_left, rectified_right, -64, 128, 5, 4, 5, 200, 2)

imgutils.show_img_grayscale(disparity_SGBM)'''

"""START TRYING TO WORK ON 4 different angles (360 degrees) TO GENERATE FULL 3D model"""

with open("dashboard/assets/Minecraft360_front.json") as fp:
    camera_parameters = json.load(fp)
    baseline = camera_parameters["baseline"]
    fov = camera_parameters["fov"]

front_L_bgr = cv2.imread("dashboard/assets/Minecraft360_front_L.png")
front_R_bgr = cv2.imread("dashboard/assets/Minecraft360_front_R.png")
front_L = cv2.cvtColor(front_L_bgr, cv2.COLOR_BGR2GRAY)
front_R = cv2.cvtColor(front_R_bgr, cv2.COLOR_BGR2GRAY)

back_L_bgr = cv2.imread("dashboard/assets/Minecraft360_back_L.png")
back_R_bgr = cv2.imread("dashboard/assets/Minecraft360_back_R.png")
back_L = cv2.cvtColor(back_L_bgr, cv2.COLOR_BGR2GRAY)
back_R = cv2.cvtColor(back_R_bgr, cv2.COLOR_BGR2GRAY)

left_L_bgr = cv2.imread("dashboard/assets/Minecraft360_left_L.png")
left_R_bgr = cv2.imread("dashboard/assets/Minecraft360_left_R.png")
left_L = cv2.cvtColor(left_L_bgr, cv2.COLOR_BGR2GRAY)
left_R = cv2.cvtColor(left_R_bgr, cv2.COLOR_BGR2GRAY)

right_L_bgr = cv2.imread("dashboard/assets/Minecraft360_right_L.png")
right_R_bgr = cv2.imread("dashboard/assets/Minecraft360_right_R.png")
right_L = cv2.cvtColor(right_L_bgr, cv2.COLOR_BGR2GRAY)
right_R = cv2.cvtColor(right_R_bgr, cv2.COLOR_BGR2GRAY)

# rectified_front_L, rectified_front_R = rectify.rectify(front_L, front_R)
disparity_front = dp.disparity_simple(front_L, front_R, -64, 128, 5, 4, 5, 200, 2)

# rectified_back_L, rectified_back_R = rectify.rectify(back_L, back_R)
disparity_back = dp.disparity_simple(back_L, back_R, -64, 128, 5, 4, 5, 200, 2)

# rectified_left_L, rectified_left_R = rectify.rectify(left_L, left_R)
disparity_left = dp.disparity_simple(left_L, left_R, -64, 128, 5, 4, 5, 200, 2)

# rectified_right_L, rectified_right_R = rectify.rectify(right_L, right_R)
disparity_right = dp.disparity_simple(right_L, right_R, -64, 128, 5, 4, 5, 200, 2)
height, width = front_L.shape
# calculate the focal length in pixels
focal_length = (width * 0.5) / np.tan(fov * 0.5 * np.pi / 180)

disparity_front = disparity_front.astype(float)
disparity_front[disparity_front == 0] = np.inf
disparity_front[disparity_front < 90] = 255
disparity_back = disparity_back.astype(float)
disparity_back[disparity_back == 0] = np.inf
disparity_back[disparity_back < 90] = 255
disparity_left = disparity_left.astype(float)
disparity_left[disparity_left == 0] = np.inf
disparity_left[disparity_left < 90] = 255
disparity_right = disparity_right.astype(float)
disparity_right[disparity_right == 0] = np.inf
disparity_right[disparity_right < 90] = 255

X = np.array([i / height for i in range(width)] * height)
Y = np.array([(height - i // width) / height for i in range(height * width)])

Z_front = ((baseline * focal_length) / disparity_front).reshape(height * width)
Z_back = ((baseline * focal_length) / disparity_back).reshape(height * width)
Z_left = ((baseline * focal_length) / disparity_left).reshape(height * width)
Z_right = ((baseline * focal_length) / disparity_right).reshape(height * width)


def cutoff(Z, X, Y, cutoff_value_min, cutoff_value_max):
    keep = np.where(Z > cutoff_value_min)[0]
    X = X[keep]
    Y = Y[keep]
    Z = Z[keep]
    keep = np.where(Z < cutoff_value_max)[0]
    X = X[keep]
    Y = Y[keep]
    Z = Z[keep]
    return X, Y, Z


f_x, f_y, f_z = cutoff(Z_front, X, Y, 0.5, 1.75)
b_x, b_y, b_z = cutoff(Z_back, X, Y, 0.5, 1.75)
l_x, l_y, l_z = cutoff(Z_left, X, Y, 0.5, 1.75)
r_x, r_y, r_z = cutoff(Z_right, X, Y, 0.5, 1.75)


# width/heigh is the max X value
# Use -Z for front to fix X axis
xyz_front = np.stack((f_x, f_y, -f_z + width / height), axis=1)
xyz_left = np.stack((-l_z + width / height, l_y, l_x), axis=1)
xyz_back = np.stack((b_x, b_y, b_z), axis=1)
xyz_right = np.stack((r_z, r_y, r_x), axis=1)


total_point_cloud = np.concatenate((xyz_front, xyz_left, xyz_back, xyz_right), axis=0)
total_point_cloud = np.concatenate((total_point_cloud, np.zeros((1, 3))), axis=0)

point_cloud = open3d.geometry.PointCloud()
point_cloud.points = open3d.utility.Vector3dVector(total_point_cloud)
# point_cloud = point_cloud.voxel_down_sample(voxel_size=0.05)

grid = open3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=0.05)
open3d.visualization.draw_geometries([grid])
# imgutils.show_point_cloud(point_cloud)
