from dash import dcc

from dashboard.layout import navbar

# ToDo: Write our own text
layout = [
    navbar.layout,
    dcc.Markdown(
        """
# Photogrammetry Beginners Practical WS 2021/22
```py
print("Welcome to our Project Page")
```
`>>>"Welcome to our Project Page"`

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


![name](assets/home/logo.png)

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
    - plot resuslts

-----------------------
To get used to working with the different Python libraries and working with images, point clouds and meshes in general we first approached two tasks, which are not directly related to the goal of our practical.
These two are Image Background segmentation and creating Litophanes of images.
Afterwards we started working on our Stereo Photogrammetry approach.


### Segmentation

#### Description
Seperate the main object of an image from the image background
#### Algorithm
1. Shadow Reduction by converting the image to HSV format and turning up brightness, then back to BGR
2. Convertion to grayscale
3. Use of a fixed threshold to create a mask (make all grayscal values higher than x completly black and all others white)
4. Use openCV to smooth the edges of the mask and to remove noise
5. Find the largest contour in the mask (since this is the one most probable to cover the wanted object) --> remove all other contours
6. apply mask to image
#### Results

![name](assets/home/segmentation.png) ![name](assets/home/segmentation2.png)

### Litophane
#### Description

Litophane is an old technique. It uses a plate with engraved contours. When light passes through the plate parts with thicker contours are darker (since less light can pass).
Our application turns this approach around. We assume that darker spots of an image are located in the front of the 3D scene while brighter spots are in the back.

#### Algorithm

Using the color values directly as depth informatio

#### Results

![name](assets/home/litophane2.png) ![name](assets/home/litophane.png)

### Stereo Photogrammetry

#### Prerequisite
Having 2 Stereo Image pairs, showing the same scene which are horizontally shifted by a certain baseline.
#### 1. Rectification
To make it easier for the disparity map algorithm we performed a rectification on our images pairs to correct non perfect horizontal shifts.
When the image pairs are perfectly shifted and therefore lay on the same plane the epipolar lines, as explained in the Theory, are perfectly horizontal and therefore finding the same pixels on both images to calculate the disparity between them is reduced to a search on the epipolar line.
To do so we used a SIFT keypoint matching algorithm by OpenCV, can find some very unique points on both images and match them (it is important to note that this keypoint matching can not be used later for the disparity map since it requires all pixles to be matched and the SIFT keypoint matcher can only find certain ones, so it is sensible to rectify images so the later pixel matching is easier).
We then calculated the epipolar lines between the found keypoints. Now the images need to be transformed until these are perfetly horizontal. Then the image is rectified
##### Results


#### 2. Disparity Map
We used the openCV StereoBSGM algorithm to perform a block/pixel matching between our rectified image pairs and to calculate the displacement vector between those blocks/pixels --> disparity map.
##### Problems when calculating disparity map
- need of patterns on objects
- baseline shift between image pairs needs to be chosen so that an parallax effect can be seen
- requires good depth of field
##### Results

#### 3. Stereo Point Cloud
The depth of every pixel in the image can be recovered  of the disparity map by INSERT FORMULA as explained in the theory.

##### Results

#### 3D Model / Mesh
To create a full 3D Mesh out of our Point clouds we need to calculate such a point cloud for every side of an object (object from the left, right, top etc.)
Then these different Models from different perspectives need to be matched togheter, which we did not further study in our project, but which could probably again be done by some sort of keypoint matching.

##### Results



""",
        highlight_config={"theme": "dark"},
        style={"margin": "0 auto", "width": "50%", "textAlign": "start"},
    ),
]
