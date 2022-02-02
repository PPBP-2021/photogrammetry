import pathlib
from typing import Callable
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple

import cv2
import dash
import numpy as np
import plotly.express as px
from dash import dcc
from dash import html

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
import modelbuilder.litophane
from dashboard.instance import app
from dashboard.layout import image_picker
from dashboard.layout import navbar
from dashboard.layout import stereo_properties
from image_utils import triangle_mesh_to_fig

# all different PROPERTIES that are used to calc the Disparity
PROPERTIES: List[str] = [
    "minDisparity",
    "numDisparities",
    "window_size",
    "disp12MaxDiff",
    "uniquenessRatio",
    "speckleWindowSize",
    "speckleRange",
    "preFilterCap",
]

IMG_LEFT: Optional[np.ndarray] = None  # left image of the stereo pair
IMG_RIGHT: Optional[np.ndarray] = None  # right image of the stereo pai
IMG_PATH: Optional[str] = None  # image path of the left image

# list of all PROPERTIES values used to call the Disparity calc function
PROPERTY_VALS: List[int] = [0, 5 * 16, 5, 12, 10, 50, 5, 63]


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


def calculate_stereo_litophane(
    asset_images: List[Tuple[pathlib.Path, pathlib.Path, dict]],
    resolution: float,
    z_scale: Callable[[float], float] = lambda z: z,
):
    """Calculate the current stereo_litophane to be shown on the website togheter with its disparity map

    Parameters
    ----------
    asset_images : List[Tuple[pathlib.Path, pathlib.Path, dict]]
        All possible image pair path Triples to choose from.
    resolution : float
        Change resolution of the image to save computation time
    Optional[z_scale] : Callable[[float], float]
        The z_scale used for the litophane

    Returns
    -------
    Tuple[List[str], List[go.Figure]]
        The Figures to be shown on the website with their according titles
    """

    # extract image information
    for image in asset_images:
        if cast(str, IMG_PATH) in str(image[0]):
            baseline = float(image[2]["baseline"])
            fov = float(image[2]["fov"])

    # resize image
    img_left = cv2.resize(IMG_LEFT, (0, 0), fx=resolution, fy=resolution)
    img_right = cv2.resize(IMG_RIGHT, (0, 0), fx=resolution, fy=resolution)

    # feature matching
    left_points, right_points, _ = modelbuilder.litophane.match_keypoints(
        img_left, img_right  # type: ignore
    )

    # calculate disparity map
    disparity = modelbuilder.litophane.calculate_disparity(
        left_points,
        right_points,
        img_left,  # type: ignore
        img_right,  # type: ignore
        *PROPERTY_VALS
    )

    # calculate the stereo_litophane
    lito_mesh = modelbuilder.litophane.calculate_stereo_litophane_mesh(
        disparity, baseline, fov, z_scale
    )
    lito_fig = triangle_mesh_to_fig(lito_mesh)

    # create figures to show on website
    titles = ["Disparity Map", "Stereo Litophane"]
    figures = [
        px.imshow(disparity, color_continuous_scale="gray").update_layout(
            margin=dict(b=0, l=0, r=0, t=0)
        ),
        lito_fig,
    ]

    return titles, figures


layout = [
    image_picker.layout,
    stereo_properties.layout,
    navbar.layout,
    html.Div(
        dcc.Loading(html.Div([], id="graphs-out-stereo")),
        style={"margin": "0 auto", "width": "50%", "textAlign": "start"},
    ),
]


@app.callback(
    dash.Output("graphs-out-stereo", "children"),
    [dash.Input(image_id[0].stem, "n_clicks") for image_id in assets.get_asset_images()]
    + [dash.Input(prop, "value") for prop in PROPERTIES]
    + [dash.Input("resolution_stereo", "value"), dash.Input("z_scale_stereo", "value")],
)
def select_image(*inputs):
    global IMG_LEFT
    global IMG_RIGHT
    global PROPERTY_VALS

    asset_images = assets.get_asset_images()
    # update all our property values
    PROPERTY_VALS = cast(List[int], inputs[len(asset_images) : -2])  # typing related
    resolution: float = cast(float, inputs[-2])
    radio_z_scale_choice: str = cast(str, inputs[-1])

    ctx = dash.callback_context
    prop_id = ctx.triggered[0]["prop_id"]

    # Only image button has .n_clicks property.
    # Thus if its still the same as before, it was only a value change
    is_property = prop_id.replace(".n_clicks", "") == prop_id

    # no current image selected, but input values changed. -> Do nothing
    if is_property and IMG_LEFT is None:
        return

    # The input that triggered this callback was the change of an image
    elif not is_property:
        # inside the buttons id we stored its asset path, thus remove nclicks
        image_path = prop_id.replace(".n_clicks", "")
        _update_selected_images(image_path, asset_images)

    titles, figures = calculate_stereo_litophane(
        asset_images, resolution, _select_scaling(radio_z_scale_choice)
    )

    return graphs.create_graph_card_vertical(titles, figures)


def _update_selected_images(
    image_path: str, assets: List[Tuple[pathlib.Path, pathlib.Path, dict]]
):
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
    global IMG_PATH
    for image in assets:
        if image_path in str(image[0]):
            IMG_LEFT = cv2.cvtColor(cv2.imread(str(image[0])), cv2.COLOR_BGR2GRAY)
            IMG_RIGHT = cv2.cvtColor(cv2.imread(str(image[1])), cv2.COLOR_BGR2GRAY)
            IMG_PATH = image_path
            break
