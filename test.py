import imageprocessing as imgp
import image_utils as imgutils
import modelbuilder as mb
import cv2

import math


stereo_left_img = cv2.imread("testimages/monke_L.png")
stereo_right_img = cv2.imread("testimages/monke_R.png")
stereo_litophane = mb.litophane_from_stereo(stereo_left_img,
                                            stereo_right_img,
                                            baseline=0.065,
                                            focal_length=0.05,
                                            fov=39.5978,
                                            resolution=1,
                                            z_scale=lambda z: z,
                                            match_features=True)
# imgutils.show_stl(stereo_litophane)
