import imageprocessing as imgp
import image_utils as imgutils
import modelbuilder as mb

import math


seg_img = imgp.segmentate_grayscale(
    r"testimages/man.jpg", 240, explain=False)
# imgutils.show_img(seg_img)
litograph = mb.litograph_from_image(seg_img,
                                    resolution=0.5,
                                    z_scale=lambda z: z/20)
litograph.save("litograph.stl")
