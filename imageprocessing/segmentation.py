from typing import Union
import cv2
import numpy as np

import image_utils as imgutils


def segmentate_grayscale(image: Union[np.ndarray, str], threshold: float, explain: bool = False) -> np.ndarray:
    """Segmentates the given image by the given threshold.

    Every pixel that has a higher grayscale value than the given threshold
    will be turned completely black.

    Parameters
    ----------
    image : Union[np.ndarray, str]
        Takes either a np.ndarray holding all pixels in BGR format or a String of the images name.
    threshold : float
        Grayscaled pixels with a higher value than this will be turned completely black.

    Returns
    -------
    np.ndarray
        np.ndarray holding all pixels in BGR format segmentated by the threshold.
    """

    if isinstance(image, str):
        image = cv2.imread(image)

    # Step 1: Try to reduce shadows
    # start by splitting image into HSV channels
    _hue, _sat, _value = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    # turn up the value to bright
    _value[:] = 200
    # merge our channels to one image
    hsv_img = cv2.merge((_hue, _sat, _value))
    bgr_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    if explain:
        imgutils.show_img(bgr_img, title="Reduced Shadows")

    # Step 2: Grayscale and normalize
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.normalize(gray_img, gray_img, 0, 255, cv2.NORM_MINMAX)
    if explain:
        imgutils.show_img(gray_img, title="Grayscaled")

    # Step 3: Remove high frequency edges
    blurred_img = cv2.GaussianBlur(gray_img, (0, 0), 3, 3)
    # apply morphology to remove isolated extraneous noise
    # use borderconstant of black since foreground touches the edges
    kernel = np.ones((3, 3), np.uint8)
    blurred_img = cv2.morphologyEx(blurred_img, cv2.MORPH_OPEN, kernel)
    denoised = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)
    mask = cv2.threshold(denoised, thresh=threshold,
                         maxval=255, type=cv2.THRESH_BINARY_INV)[1]
    if explain:
        imgutils.show_img(mask, title="Mask")

    # Step 4: Find largest contour
    largest_cntr = np.zeros(image.shape, dtype=image.dtype)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    largest_cntr_indx = max(enumerate(contours),
                            key=lambda x: cv2.contourArea(x[1]))[0]
    bounding_rect = cv2.boundingRect(contours[largest_cntr_indx])

    largest_cntr = cv2.drawContours(image=largest_cntr,
                                    contours=contours,
                                    contourIdx=largest_cntr_indx,
                                    color=255,
                                    thickness=cv2.FILLED,
                                    hierarchy=hierarchy)

    largest_cntr = cv2.cvtColor(largest_cntr, cv2.COLOR_BGR2GRAY)

    if explain:
        imgutils.show_img(largest_cntr, title="Largest Contour")

    image = cv2.bitwise_and(image, image, mask=largest_cntr)
    x,y,w,h = bounding_rect
    image = image[y:y+h,x:x+w]

    return image
