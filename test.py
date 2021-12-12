import imageprocessing as imgp
import image_utils as imgutils
import modelbuilder as mb


seg_img = imgp.segmentate_grayscale(
    r"testimages/man.jpg", 240, explain=False)
#imgutils.show_img(seg_img)
litograph = mb.litograph_from_image(seg_img, 0.1)
litograph.save("litograph.stl")