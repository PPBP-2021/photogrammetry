import os
from typing import List

import cv2
import dash
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
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

    segmentated = segmentate_grayscale(
        file_path, 240)  # ToDo 240 to slider value?

    lito = litophane_from_image(
        segmentated,
        resolution=1.0,
        z_scale=lambda z: z
    )

    vertices = lito.vectors[:, :]
    x, y, z = vertices

    mesh3D = go.Mesh3d(x=x,
                       y=y,
                       z=z,
                       flatshading=True,
                       # colorscale=colorscale,
                       intensity=z,
                       # name=title,
                       showscale=False)

    layout = go.Layout(paper_bgcolor='rgb(1,1,1)',
                       # title_text=title, title_x=0.5,
                       font_color='white',
                       scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=False,
                       scene=dict(
                           aspectmode='data'
                       ))
    fig = go.Figure(data=[mesh3D], layout=layout)

    figures = [fig]

    return graphs.create_graph_card_vertical(titles, figures)
