import imageprocessing as imgp
import image_utils as imgutils
import modelbuilder as mb
import cv2

import math


#seg_img = imgp.segmentate_grayscale(
   # r"testimages/man.jpg", 240, explain=False)

#imgutils.show_img(seg_img)
#litophane = mb.litophane_from_image(seg_img,
                                   # resolution=0.5,
                                    # z_scale=lambda z: z/20)

#imgutils.show_stl(litophane)
#litophane.save("litophane.stl")

stereo_left_img = cv2.imread("testimages/im0e2.png")
stereo_right_img = cv2.imread("testimages/im1e2.png")
stereo_litophane = mb.litophane_from_stereo(stereo_left_img, stereo_right_img, 111.53, 1758.23, 69, resolution=0.2)
imgutils.show_stl(stereo_litophane)
