from typing import Any
from typing import Callable
from typing import Optional

import cv2
import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc
from dash import html

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
from dashboard.instance import app
from dashboard.layout import image_picker_stereo
from dashboard.layout import litophane_properties
from dashboard.layout import navbar
from image_utils import triangle_mesh_to_fig
from imageprocessing import segmentate_grayscale
from modelbuilder import litophane_from_image


def _create_segmentation_fig(seg_img):

    seg_fig = px.imshow(cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)).update_layout(
        template="plotly_white",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin=dict(
            b=0,  # bottom margin 40px
            l=0,  # left margin 40px
            r=0,  # right margin 20px
            t=0,  # top margin 20px
        ),
    )

    return seg_fig


layout = [
    dcc.Store(id="memory-lito", storage_type="session"),
    image_picker_stereo.layout,  # the image picker on the very left side
    litophane_properties.layout,  # properties on the very right side
    navbar.layout,  # navigation on top of the website
    html.Div(
        dcc.Loading(html.Div([], id="graphs-out-lito")),
        style={
            "margin": "0 auto",
            "width": "50%",
            "textAlign": "start",
        },  # centered and 50% of width
    ),
]


def _select_scaling(radio_choice: str) -> Callable[[float], float]:
    """Get the corresponding math function for the given string.

    Parameters
    ----------
    radio_choice : str
        The selected radio button in the web ui.

    Returns
    -------
    Callable[float, float]
        The function to be used as z scale for the mesh.
    """
    return {
        "log": np.log,
        "quadratic": np.square,
    }.get(radio_choice, lambda z: z)


def _update_segmentation(treshold, asset_images=None, image_path=None):
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

    for image in asset_images:
        if image_path in str(image[0]):
            image_path = str(image[0])

    return segmentate_grayscale(image_path, treshold)


@app.callback(
    dash.Output("graphs-out-lito", "children"),
    dash.Output("memory-lito", "data"),
    inputs={
        "image_buttons": [
            dash.Input(image_id[0].stem, "n_clicks")
            for image_id in assets.get_asset_images_stereo()
        ],
        "gray_treshold": dash.Input("gray_treshold", "value"),
        "resolution": dash.Input("resolution", "value"),
        "z_scale": dash.Input("z_scale", "value"),
    },
    state={"memory": dash.State("memory-lito", "data")},
)
def select_image(
    image_buttons: list,
    gray_treshold: float,
    resolution: float,
    z_scale: str,
    memory: dict,
):

    if memory is None:
        memory = {}

    asset_images = assets.get_asset_images_stereo()

    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    selected_image: Optional[str] = memory.get("selected_image", None)
    image_path: Optional[str] = selected_image

    # Only image button has .n_clicks property.
    # Thus if its still the same as before, it was only a value change
    is_property = prop_id.replace(".n_clicks", "") == prop_id

    # no current image selected, but properties changed. -> Do nothing
    if is_property and selected_image is None:
        return None, {}

    # The input that triggered this callback was the change of an image
    elif not is_property:
        # inside the buttons id we stored its asset path, thus remove nclicks
        image_path = prop_id.replace(".n_clicks", "")
        gray_seg = _update_segmentation(gray_treshold, asset_images, image_path)

    # The input that triggered this callback was a property change
    else:
        prop = prop_id.replace(".value", "")
        # The input property was the threshold for our grayscale
        if prop == "gray_treshold":
            gray_seg = _update_segmentation(gray_treshold, asset_images, image_path)

    seg_fig = _create_segmentation_fig(gray_seg)
    lito_mesh = litophane_from_image(
        gray_seg,
        resolution=resolution,
        z_scale=_select_scaling(z_scale),
    )
    lito_fig = triangle_mesh_to_fig(lito_mesh)
    titles, figures = ["Segmentated", "3D Litophane"], [seg_fig, lito_fig]

    memory["selected_image"] = image_path
    return graphs.create_graph_card_vertical(titles, figures), memory
