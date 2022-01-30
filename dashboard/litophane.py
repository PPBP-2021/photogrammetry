import os
from typing import List

import cv2
import dash
import numpy as np
import open3d
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from image_utils import triangle_mesh_to_fig
from imageprocessing import segmentate_grayscale
from modelbuilder import litophane_from_image

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
from dashboard.instance import app
from dashboard.layout import image_picker, navbar

layout = [
    image_picker.layout,  # the image picker on the very left side
    navbar.layout,  # navigation on top of the website
    html.Div(
        dcc.Loading(
            html.Div([
            ], id="graphs-out-lito")
        ), style={"margin": "0 auto", "width": "50%", "textAlign": "start"}  # centered and 50% of width
    )
]


@app.callback(dash.Output("graphs-out-lito", "children"),
              [dash.Input(image_id[0].stem, "n_clicks") for image_id in assets.get_asset_images()])
def select_image(*image_path):

    ctx = dash.callback_context
    image_path = ctx.triggered[0]["prop_id"].replace(".n_clicks", "")

    file_path = ""
    for image in assets.get_asset_images():
        if image_path in str(image[0]):
            file_path = str(image[0])
            break

    titles = ["3d Litophane", ]

    segmentated = segmentate_grayscale(file_path,
                                       240)  # ToDo 240 to slider value?

    lito: open3d.geometry.TriangleMesh = litophane_from_image(
        segmentated,
        resolution=0.1,
        z_scale=lambda z: z
    )

    fig = triangle_mesh_to_fig(lito)

    figures = [fig]

    return graphs.create_graph_card_vertical(titles, figures)
