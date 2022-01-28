import pathlib
from typing import List, Optional, Tuple, cast

import cv2
import dash
import modelbuilder.litophane
import numpy as np
import plotly.express as px
from dash import dcc, html

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
from dashboard.instance import app
from dashboard.layout import image_picker, navbar, stereo_properties

# all different PROPERTIES that are used to calc the Disparity
PROPERTIES: List[str] = ["minDisparity", "numDisparities", "window_size", "disp12MaxDiff",
                         "uniquenessRatio", "speckleWindowSize", "speckleRange", "preFilterCap"]

IMG_LEFT: Optional[np.ndarray] = None  # left image of the stereo pair
IMG_RIGHT: Optional[np.ndarray] = None  # right image of the stereo pai

# list of all PROPERTIES values used to call the Disparity calc function
PROPERTY_VALS: List[int] = [0, 5*16, 5, 12, 10, 50, 5, 63]


def calculate_current_disparity():
    left_points, right_points, _ = modelbuilder.litophane.match_keypoints(
        IMG_LEFT, IMG_RIGHT  # type: ignore
    )

    disparity = modelbuilder.litophane.calculate_disparity(
        left_points,
        right_points,
        IMG_LEFT,  # type: ignore
        IMG_RIGHT,  # type: ignore
        *PROPERTY_VALS
    )

    titles = ["Disparity Map"]
    figures = [
        px.imshow(disparity, color_continuous_scale="gray").update_layout(
            margin=dict(b=0, l=0, r=0, t=0)
        )
    ]

    return titles, figures

CONTENT_STYLE = {
    "position": "fixed",
    "top": 58,
    "left": 250,
    "bottom": 0,
    "width": "55%",
    "padding": "4rem 1rem 2rem",
    "background-color": "#f8f9fa",
}

layout = [
    image_picker.layout,
    stereo_properties.layout,
    navbar.layout,
    dcc.Loading(
        html.Div([

        ], id="graphs-out-stereo",
        style=CONTENT_STYLE)
    )
]


@app.callback(
    dash.Output("graphs-out-stereo", "children"),
    [
        dash.Input(image_id[0].stem, "n_clicks")
        for image_id in assets.get_asset_images()
    ]
    +
    [
        dash.Input(prop, "value") for prop in PROPERTIES
    ]
)
def select_image(*inputs):
    global IMG_LEFT
    global IMG_RIGHT
    global PROPERTY_VALS

    asset_images = assets.get_asset_images()
    # update all our property values
    PROPERTY_VALS = cast(
        List[int], inputs[len(asset_images):])  # typing related

    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    # Only image button has .n_clicks property.
    # Thus if its still the same as before, it was only a value change
    is_value_slider = prop_id.replace(".n_clicks", "") == prop_id

    # no current image selected, but input values changed. -> Do nothing
    if is_value_slider and IMG_LEFT is None:
        return

    # The input that triggered this callback was the change of an image
    elif not is_value_slider:
        # inside the buttons id we stored its asset path, thus remove nclicks
        image_path = prop_id.replace(".n_clicks", "")
        _update_selected_images(image_path, asset_images)

    titles, figures = calculate_current_disparity()

    return graphs.create_graph_card_vertical(titles, figures)


def _update_selected_images(image_path: str, assets: List[Tuple[pathlib.Path, pathlib.Path, dict]]):
    """Update the current selected stereo image pairs from the given input.

    Parameters
    ----------
    image_path : str
        The selected image path
    assets : List[Tuple[pathlib.Path, pathlib.Path, dict]]
        All possible image pair path Triples to choose from.
    """
    global IMG_LEFT
    global IMG_RIGHT
    for image in assets:
        if image_path in str(image[0]):
            IMG_LEFT = cv2.cvtColor(
                cv2.imread(str(image[0])), cv2.COLOR_BGR2GRAY)
            IMG_RIGHT = cv2.cvtColor(
                cv2.imread(str(image[1])), cv2.COLOR_BGR2GRAY)
            break
