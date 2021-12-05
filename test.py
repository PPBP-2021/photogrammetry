import imageprocessing as imgp
import image_utils as imgutils


import cv2
import plotly.express as px

seg_img = imgp.segmentate_grayscale(
    r"testimages/man.jpg", 240, explain=True)
imgutils.show_img(seg_img)
