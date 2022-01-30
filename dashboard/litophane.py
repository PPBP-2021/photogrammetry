from typing import List, Optional, Tuple, cast

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
from dashboard.layout import image_picker, litophane_properties, navbar

CURRENT_SEG = None
CURRENT_PATH = None


def _create_segmentation_fig(seg_img):

    seg_fig = px.imshow(
        cv2.cvtColor(
            seg_img,
            cv2.COLOR_BGR2RGB
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
        )
    )

    return seg_fig


layout = [
    image_picker.layout,  # the image picker on the very left side
    litophane_properties.layout,  # properties on the very right side
    navbar.layout,  # navigation on top of the website
    html.Div(
        dcc.Loading(
            html.Div([
            ], id="graphs-out-lito")
        ), style={"margin": "0 auto", "width": "50%", "textAlign": "start"}  # centered and 50% of width
    )
]


def _select_scaling(function):
    """returns the mathematical function for a given function name

    Parameters
    ----------
    function : str
        The name of the mathematical function
    """
    if function == "log":
        return lambda z: np.log(z)
    elif function == "quadratic":
        return lambda z: np.square(z)
    elif function == "no":
        return lambda z: z


def _update_grayscale(treshold, asset_images=None, image_path=None):
    """Update the current grayscale image with new treshold or optionally new image

    Parameters
    ----------
    treshold : float
        The treshold used for segmentate_grayscale
    image_path : Optional[str]
        The selected image path, if not given takes previously selected image
    assets : Optional[List[Tuple[pathlib.Path, pathlib.Path, dict]]]
        All possible image pair path Triples to choose from.
    """
    global CURRENT_SEG
    global CURRENT_PATH

    # if this function gets no new image path it takes the saved one
    if not image_path:
        image_path = CURRENT_PATH
    else:
        for image in asset_images:
            if image_path in str(image[0]):
                image_path = str(image[0])

    CURRENT_SEG = segmentate_grayscale(image_path, treshold)
    CURRENT_PATH = image_path


@app.callback(
    dash.Output("graphs-out-lito", "children"),
    [
        dash.Input(image_id[0].stem, "n_clicks")
        for image_id in assets.get_asset_images()
    ]
    +
    [
        dash.Input("gray_treshold", "value"),
        dash.Input("resolution", "value"),
        dash.Input("z_scale", "value")
    ]
)
def select_image(*inputs):
    global CURRENT_SEG

    asset_images = assets.get_asset_images()

    # update all our property values
    current_treshold = inputs[-3]
    current_resolution = inputs[-2]
    z_scale = inputs[-1]

    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    # Only image button has .n_clicks property.
    # Thus if its still the same as before, it was only a value change
    is_property = prop_id.replace(".n_clicks", "") == prop_id

    # no current image selected, but properties changed. -> Do nothing
    if is_property and CURRENT_PATH is None:
        return

    # The input that triggered this callback was the change of an image
    elif not is_property:
        # inside the buttons id we stored its asset path, thus remove nclicks
        image_path = prop_id.replace(".n_clicks", "")
        _update_grayscale(current_treshold, asset_images, image_path)

    # The input that triggered this callback was a property change
    else:
        prop = prop_id.replace(".value", "")
        # The input property was the threshold for our grayscale
        if prop == "gray_treshold":
            _update_grayscale(current_treshold)

    seg_fig = _create_segmentation_fig(CURRENT_SEG)
    lito_mesh = litophane_from_image(
        CURRENT_SEG, resolution=current_resolution, z_scale=_select_scaling(z_scale))
    lito_fig = triangle_mesh_to_fig(lito_mesh)
    titles, figures = ["Segmentated", "3D Litophane"], [seg_fig, lito_fig]

    return graphs.create_graph_card_vertical(titles, figures)
