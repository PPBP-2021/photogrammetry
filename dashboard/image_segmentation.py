import os
from typing import List

import cv2
import dash
import plotly.express as px
from dash import dcc
from dash import html

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
from dashboard.instance import app
from dashboard.layout import image_picker_stereo
from dashboard.layout import navbar
from imageprocessing import segmentate_grayscale


layout = [
    image_picker_stereo.layout,  # the image picker on the very left side
    navbar.layout,  # navigation on top of the website
    html.Div(
        dcc.Loading(html.Div([], id="graphs-out")),
        style={
            "margin": "0 auto",
            "width": "50%",
            "textAlign": "start",
        },  # centered and 50% of width
    ),
]


@app.callback(
    dash.Output("graphs-out", "children"),
    [
        dash.Input(image_id[0].stem, "n_clicks")
        for image_id in assets.get_asset_images_stereo()
    ],
)
def select_image(*image_path):

    ctx = dash.callback_context
    image_path = ctx.triggered[0]["prop_id"].replace(".n_clicks", "")

    file_path = ""
    for image in assets.get_asset_images_stereo():
        if image_path in str(image[0]):
            file_path = str(image[0])
            break

    titles = ["Before", "Segmentated"]
    figures = [
        px.imshow(cv2.cvtColor(cv2.imread(file_path), cv2.COLOR_BGR2RGB)).update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(
                b=0,  # bottom margin 40px
                l=0,  # left margin 40px
                r=0,  # right margin 20px
                t=0,  # top margin 20px
            ),
        ),
        px.imshow(
            cv2.cvtColor(segmentate_grayscale(file_path, 240), cv2.COLOR_BGR2RGB)
        ).update_layout(
            template="plotly_white",
            plot_bgcolor="rgba(0, 0, 0, 0)",
            paper_bgcolor="rgba(0, 0, 0, 0)",
            margin=dict(
                b=0,  # bottom margin 40px
                l=0,  # left margin 40px
                r=0,  # right margin 20px
                t=0,  # top margin 20px
            ),
        ),
    ]

    return graphs.create_graph_card_vertical(titles, figures)
