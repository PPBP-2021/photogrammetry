from typing import List
from typing import Optional

import cv2
import dash
import plotly.express as px
from dash import dcc
from dash import html

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
from dashboard.instance import app
from dashboard.layout import image_picker_segmentate
from dashboard.layout import navbar
from dashboard.layout import segmentation_properties
from imageprocessing import segmentate_grayscale


layout = [
    dcc.Store(id="memory-segmentate", storage_type="session"),
    image_picker_segmentate.layout,  # the image picker on the very left side
    segmentation_properties.layout,  # the properties on the right side
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
    output=[
        dash.Output("graphs-out", "children"),
        dash.Output("memory-segmentate", "data"),
    ],
    inputs={
        "image_buttons": [
            dash.Input(image_id[0].stem, "n_clicks")
            for image_id in assets.get_asset_images_segmentate()
        ],
        "grayscale_threshold": dash.Input("grayscaleThreshold", "value"),
    },
    state={"memory": dash.State("memory-segmentate", "data")},
)
def select_image(image_buttons: list, grayscale_threshold: int, memory: dict):
    if memory is None:
        memory = {}

    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    # Only image button has .n_clicks property.
    is_property = prop_id.replace(".n_clicks", "") == prop_id

    selected_image: Optional[str] = memory.get("selected_image", None)
    image_button_name: Optional[str] = selected_image

    if is_property and selected_image is None:
        return None, {}
    elif not is_property:  # A button was clicked
        image_button_name = prop_id.replace(
            ".n_clicks", ""
        )  # get the image name from the button

    img_dict = assets.get_asset_image_dict_segmentate(
        image_button_name
    )  # get the dict with the actual image path
    file_path = img_dict["image"]
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
            cv2.cvtColor(
                segmentate_grayscale(file_path, grayscale_threshold), cv2.COLOR_BGR2RGB
            )
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

    memory["selected_image"] = image_button_name
    return graphs.create_graph_card_vertical(titles, figures), memory
