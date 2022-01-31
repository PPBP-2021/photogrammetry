from dash import dcc

from dashboard.layout import navbar

layout = [
    navbar.layout,
    dcc.Markdown("""
# Photogrammetry Beginners Practical WS 2021/22
-----------------------------------------------
```py
print("Welcome to our Project Page")
```
`>>>"Welcome to our Project Page"`

## What is Photogrammetry?

Photogrammetry is the process of creating 3D objects from 2D photographs. The process can be used to scan anything from objects and people, to architecture, terrain and landscapes. There are various strategies you can use when taking photos to produce accurate models. This document includes photography techniques, ideal camera settings and lighting, photogrammetry software, and further resources to create 3D models from photos.

### Definition

  **Photogrammetry** *[noun]* The science or technique for obtaining reliable information on the natural environment or physical objects by recording, measuring and interpreting photographic images.

  Greek:

  - *"photos"* (light)
  - *"gramma"* (something written or drawn)
  - *"metron"* (measure)

#### Typical Outputs

A map, a drawing, a 3D model of a real-world object, scene, or terrain.

#### Related fields

Remote Sensing, GIS, Stereoscopy

#### Main Tasks of Photogrammetry

  - To measure something without touching it
  - To measure something that may no longer exist, or may only exist in photographs
  - To measure something too large to measure with traditional methods, i.e., landscape, a megalithic structure
  - Quantitative data from photographs, the science of measuring in photos.

""", highlight_config={"theme": "dark"}, style={"margin": "0 auto", "width": "50%", "textAlign": "start"})]
