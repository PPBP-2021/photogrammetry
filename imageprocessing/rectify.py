from typing import cast
from typing import Tuple

import cv2
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots

import image_utils as imgutils

# ToDo: Add own comments/explanations, at least for presentation purposes


def rectify(
    img_l: np.ndarray, img_r: np.ndarray, *, explain: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rectify the images without a calibration matrix.
    Args:
        img_l: numpy array of the left image
        img_r: numpy array of the right image
        explain: if True, show images of the algorithm steps

    Returns:
        img_l_rect: numpy array of the left image rectified
        img_r_rect: numpy array of the right image rectified
    """

    # Use SIFT to find keypoints and descriptors.
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img_l, None)
    kp_r, des_r = sift.detectAndCompute(img_r, None)

    if explain:
        # Visualize keypoints
        img_sift = cv2.drawKeypoints(
            img_l, kp_l, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        imgutils.show_img(img_sift, title="SIFT keypoints")

    # Match keypoints in both images
    # Based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_l, des_r, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    matches_mask = [[0, 0] for i in range(len(matches))]
    pts_l = []
    pts_r = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            # Keep this keypoint pair
            matches_mask[i] = [1, 0]
            pts_r.append(kp_r[m.trainIdx].pt)
            pts_l.append(kp_l[m.queryIdx].pt)

    if explain:
        # Draw the keypoint matches between both pictures
        # Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matches_mask[300:500],
            flags=cv2.DrawMatchesFlags_DEFAULT,
        )

        keypoint_matches = cv2.drawMatchesKnn(
            img_l, kp_l, img_r, kp_r, matches[300:500], None, **draw_params
        )

        imgutils.show_img(keypoint_matches)

    # ------------------------------------------------------------
    # STEREO RECTIFICATION

    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
    pts_l = cast(np.ndarray, np.int32(pts_l))
    pts_r = cast(np.ndarray, np.int32(pts_r))
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts_l, pts_r, cv2.FM_RANSAC)

    # We select only inlier points
    pts_l = cast(np.ndarray, pts_l[inliers.ravel() == 1])
    pts_r = cast(np.ndarray, pts_r[inliers.ravel() == 1])

    if explain:
        # Visualize epilines
        # Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
        def drawlines(img1src, img2src, lines, pts1src, pts2src):
            """img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines"""
            r, c = img1src.shape
            img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
            img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
            # Edit: use the same random seed so that two images are comparable!
            np.random.seed(0)
            for r, pt1, pt2 in zip(lines, pts1src, pts2src):
                color = tuple(np.random.randint(0, 255, 3).tolist())
                x0, y0 = map(int, [0, -r[2] / r[1]])
                x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
                img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
                img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
                img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
            return img1color, img2color

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(
            cast(np.ndarray, pts_r).reshape(-1, 1, 2), 2, fundamental_matrix
        )
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(img_l, img_r, lines1, pts_l, pts_r)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(
            cast(np.ndarray, pts_l).reshape(-1, 1, 2), 1, fundamental_matrix
        )
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(img_r, img_l, lines2, pts_r, pts_l)

        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(px.imshow(img5).data[0], row=1, col=1)
        fig.add_trace(px.imshow(img3).data[0], row=1, col=2)
        fig.show()

    # Stereo rectification (uncalibrated variant)
    # Adapted from: https://stackoverflow.com/a/62607343
    h_l, w_l = img_l.shape
    h_r, w_r = img_r.shape
    _, H_l, H_r = cv2.stereoRectifyUncalibrated(
        np.float32(pts_l), np.float32(pts_r), fundamental_matrix, imgSize=(w_l, h_l)
    )

    # Undistort (rectify) the images and save them
    # Adapted from: https://stackoverflow.com/a/62607343
    img_l_rectified = cv2.warpPerspective(img_l, H_l, (w_l, h_l))
    img_r_rectified = cv2.warpPerspective(img_r, H_r, (w_r, h_r))

    if explain:
        cv2.imwrite("rectified_1.png", img_l_rectified)
        cv2.imwrite("rectified_2.png", img_r_rectified)

        # Draw the rectified images
        fig = make_subplots(rows=1, cols=2)
        fig.add_trace(px.imshow(img_l_rectified).data[0], row=1, col=1)
        fig.add_trace(px.imshow(img_r_rectified).data[0], row=1, col=2)
        fig.show()

    return img_l_rectified, img_r_rectified
