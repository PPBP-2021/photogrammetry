import imageprocessing as imgp
import image_utils as imgutils


seg_img = imgp.segmentate_grayscale(
    r"testimages/man.jpg", 240, explain=True)
imgutils.show_img(seg_img)
