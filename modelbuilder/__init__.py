import stl
import numpy as np
from stl.mesh import Mesh
import cv2
import image_utils


from imageprocessing.segmentation import segmentate_grayscale


def litograph_from_image(seg_img: np.ndarray, resolution: float):
    """Construct a 3D litograph of a segmentated image.

    Parameters
    ----------
    seg_img : np.ndarray
        Image segmentated into foreground, background. Background is completly black.
    
    resolution : float
        Value between 0 and 1; the percentage of pixels used.
    """
    seg_img = cv2.resize(seg_img, (0,0),fx=resolution,fy=resolution)
    seg_img = cv2.cvtColor(seg_img,cv2.COLOR_BGR2GRAY)
    height, width = seg_img.shape
  
    vertices = []
    faces = []

    for i,row in enumerate(seg_img):
        for j,pixel in enumerate(row):
            Z = pixel / 255
            vertices.append([j,height-i,Z])
    
    vertices = np.array(vertices)
    vertices = vertices.reshape((height,width,3))
    
    for i in range(height-1):
        for j in range(width-1):
            top_left = vertices[i][j]
            bottom_right = vertices[i+1][j+1]
            face_1 = [top_left, vertices[i+1][j], bottom_right]
            face_2 = [top_left, vertices[i][j+1], bottom_right]
            faces.append(face_1)
            faces.append(face_2)
    
    faces = np.array(faces,dtype=Mesh.dtype)
    print(faces.shape)

    return Mesh(faces)