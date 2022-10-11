from dash import dcc

from dashboard.layout import navbar

# ToDo: Write our own text
layout = [
    navbar.layout,
    dcc.Markdown(
        """
# Photogrammetry using Stereo Image Pairs
```py
print("Welcome to our Project Page")
```
`>>>"Welcome to our Project Page"`

Check this project out on [GitHub](https://github.com/PPBP-2021/photogrammetry).
Or see our [presentation](https://github.com/PPBP-2021/photogrammetry/blob/main/presentation.pdf) (german).

-----------------------------------------------

## What is Photogrammetry?
Photogrammetry is the science of making measurements from photographs.
It is used to reconstruct the third dimension (the depth) of two-dimensional images.
The application of photogrammetry is widespread and includes surveying, measuring, modeling, and 3D reconstruction.

### Related fields
- land surveying
    - google earth
- computer vision
    - autonomous driving
- computer graphics
    - 3D modeling

### Approaches
- Shape from Shading
    - reconstruct the shape of an object using shadow and light information
- Structure from Motion
    - reconstruct the 3D structure of a scene from a set of 2D images
- **Stereo-photogrammetry** (our approach)
    - keypoint triangulation

-----------------
## Motivation
In the past we worked on a video game. One of the main challenges was to create good looking 3D models.
Creating 3D models is a time consuming task.
We already knew of *photogrammetry* as a way to automate model generation, but didn't know how it worked.
Furthermore we are generally interested in *computer vision* and *image processing*.
So we decided to learn more about it and create our own photogrammetry project.


### Goals
- learn about camera basics
    - 3D to 2D projection
- learn about image processing
    - implement in python
 - generate meshes in python
    - different mesh formats

**➤ generate 3D-model from images**

-------------------------
## Theory

In its simplest form an image is the projection of a 3D scene onto a 2D plane.
This projection can be illustrated by the pinhole camera model.

![pinhole camera model](assets/home/opencv/pinhole_camera_model.png)

Such a projection can be described by the following equation:

![intrinsic extrinsic projection](assets/home/intrinsic_extrinsic_projection.png)

To get back to the 3D coordinates we need to invert the equation.
This is not fully possible from a single image, but we can get a good approximation by using multiple images.
For this we need to know the position of the camera in the 3D scene.
Furthermore we assume that our cameras are perfectly coplanar, this means that the
[epipolar lines](https://en.wikipedia.org/wiki/Epipolar_geometry) are perfectly horizontal.



![epipolar lines co planar](assets/home/epipolar_lines_co_planar.png)
![3d reconstruction](assets/home/3d_reconstruction.png)

By exploiting the epipolar geometry we can triangulate the 3D coordinates of a point from two images.


--------------------------

## Practice

#### Used modules
- Dash
    - website
- OpenCV
    - general image library
- Open3D
    - meshes and point clouds
- Plotly
    - plot results

-----------------------
To familiarize ourself with the previously listed Python libraries we first worked on two smaller algorithms,
image background segmentation and litophane generation.
These two algorithms are not directly related to our final photogrammetry process, but they helped us to get a better
understanding of image processing and mesh generation.
Afterwards we started working on our Stereo Photogrammetry approach.




### Segmentation

Separate the main object of an image from the image background
#### Algorithm
1. remove dark noise by increasing brightness in HSV format
2. conversion to grayscale
3. use of a fixed threshold to create a mask (make all grayscale values higher than x completely black and all others
white)
4. use openCV to smooth the edges of the mask and to remove noise
5. find the largest contour in the mask (since this is the one most probable to cover the wanted object) --> remove
all other contours
6. apply mask to image
#### Results

![name](assets/home/segmentation.png) ![name](assets/home/segmentation2.png)

### Litophane


Litophane is an old technique. It uses a plate with engraved contours. When light passes through the plate parts with
thicker contours are darker (since less light can pass).
Our application turns this approach around. We assume that darker spots of an image are located in the front of the
3D scene while brighter spots are in the back.

#### Algorithm

1. convert RGB value to grayscale (0-255)
2. use the grayscale value as Z coordinate

#### Results

![name](assets/home/litophane2.png) ![name](assets/home/litophane.png)

### Stereo Photogrammetry

#### Prerequisite
Having 2 Stereo Image pairs, showing the same scene which are horizontally shifted by a certain baseline.
#### 1. Rectification
To make it easier for the disparity map algorithm we performed a rectification on our images pairs to correct non
perfect horizontal shifts.
When the image pairs are perfectly shifted and therefore lie on the same plane, the epipolar lines are perfectly
horizontal.
This allows us to reduce the search for matching pixels to the epipolar line.
To do so we used a SIFT keypoint matching algorithm by OpenCV, which finds unique points on both images and
matches them (it is important to note that this keypoint matching can not be used later because the disparity map
requires all pixels to be matched, while SIFT only matches unique keypoints).
The resulting epipolar lines from the matched keypoints are used to transform the images so that the epipolar lines
are horizontally aligned.

##### Results
![rectified](assets/home/rectified.png)


#### 2. Disparity Map
We used the openCV StereoBSGM algorithm to perform a block/pixel matching between our rectified image pairs and to
calculate the displacement vector between those blocks/pixels --> disparity map.
##### Problems when calculating disparity map
- pixel/block matching requires patterns
    - it is impossible to match pixels between the images if every pixel has the same color
- baseline shift between image pairs needs to be chosen so that an parallax effect can be seen
- requires good depth of field
    - if the depth of field is too small too much noise is introduced

##### Results
![disparity](assets/home/disparity.png)

#### 3. Stereo Point Cloud
The depth of each pixel is calculated by multiplying the disparity value with the baseline and the focal length.
![z formula](assets/home/z_formula.png)

##### Results
![stereo point cloud](assets/home/stereo_point_cloud.png)

#### 3D Model / Mesh
To create a full 3D Mesh out of our Point clouds we need to calculate such a point cloud for every side of an object
(object from the left, right, top etc.)
Then these different Models from different perspectives need to be matched together, which we did not further study
in our project, but which could probably again be done by some sort of keypoint matching.

##### Results
![final model](assets/home/final_model.png)

""",
        highlight_config={"theme": "dark"},
        style={"margin": "0 auto", "width": "50%", "textAlign": "start"},
    ),
]
