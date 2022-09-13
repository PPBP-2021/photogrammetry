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

""",
        highlight_config={"theme": "dark"},
        style={"margin": "0 auto", "width": "50%", "textAlign": "start"},
    ),
]
