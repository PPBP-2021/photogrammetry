from typing import Callable
from typing import List
from typing import Optional

import cv2
import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import dcc
from dash import html

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
import modelbuilder
from dashboard.instance import app
from dashboard.layout import image_picker_final_model
from dashboard.layout import navbar
from dashboard.layout import stereo_properties
from imageprocessing import disparity as dp
from imageprocessing import rectify as rf

# all different PROPERTIES that are used to calc the Disparity maps
PROPERTIES: List[str] = [
    "minDisparity",
    "numDisparities",
    "block_size",
    "disp12MaxDiff",
    "uniquenessRatio",
    "speckleWindowSize",
    "speckleRange",
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


def update_final_model(
    img_dict: dict,
    properties: List[int],
    resolution: float,
    z_scale: Callable[[float], float] = lambda z: z,
):
    """Calculate the current 3D Model to be shown on the website

    Parameters
    ----------
    img_dict : dict
        The dictionary containing the image names and their properties.
    properties : List[int]
        The list of properties to be used for the disparity calculation.
    resolution : float
        Change resolution of the image to save computation time
    Optional[z_scale] : Callable[[float], float]
        The z_scale used for the point_cloud

    Returns
    -------
    Tuple[List[str], List[go.Figure]]
        The Figures to be shown on the website with their according titles
    """

    baseline = img_dict["baseline"]
    fov = img_dict["fov"]

    # get all stereo image pairs needed for the 3D model
    img_front_L = cv2.cvtColor(
        cv2.imread(img_dict["front_image_pair"][0]), cv2.COLOR_BGR2GRAY
    )
    img_front_R = cv2.cvtColor(
        cv2.imread(img_dict["front_image_pair"][1]), cv2.COLOR_BGR2GRAY
    )

    img_back_L = cv2.cvtColor(
        cv2.imread(img_dict["back_image_pair"][0]), cv2.COLOR_BGR2GRAY
    )
    img_back_R = cv2.cvtColor(
        cv2.imread(img_dict["back_image_pair"][1]), cv2.COLOR_BGR2GRAY
    )

    img_left_L = cv2.cvtColor(
        cv2.imread(img_dict["left_image_pair"][0]), cv2.COLOR_BGR2GRAY
    )
    img_left_R = cv2.cvtColor(
        cv2.imread(img_dict["left_image_pair"][1]), cv2.COLOR_BGR2GRAY
    )

    img_right_L = cv2.cvtColor(
        cv2.imread(img_dict["right_image_pair"][0]), cv2.COLOR_BGR2GRAY
    )
    img_right_R = cv2.cvtColor(
        cv2.imread(img_dict["right_image_pair"][1]), cv2.COLOR_BGR2GRAY
    )

    # resize image
    img_front_L = cv2.resize(img_front_L, (0, 0), fx=resolution, fy=resolution)
    img_front_R = cv2.resize(img_front_R, (0, 0), fx=resolution, fy=resolution)
    img_back_L = cv2.resize(img_back_L, (0, 0), fx=resolution, fy=resolution)
    img_back_R = cv2.resize(img_back_R, (0, 0), fx=resolution, fy=resolution)
    img_left_L = cv2.resize(img_left_L, (0, 0), fx=resolution, fy=resolution)
    img_left_R = cv2.resize(img_left_R, (0, 0), fx=resolution, fy=resolution)
    img_right_L = cv2.resize(img_right_L, (0, 0), fx=resolution, fy=resolution)
    img_right_R = cv2.resize(img_right_R, (0, 0), fx=resolution, fy=resolution)

    # rectify images
    # TODO

    # calculate all disparity maps
    disparity_front = dp.disparity_simple(
        img_front_L,  # type: ignore
        img_front_R,  # type: ignore
        *properties,
    )

    disparity_back = dp.disparity_simple(
        img_back_L,  # type: ignore
        img_back_R,  # type: ignore
        *properties,
    )

    disparity_left = dp.disparity_simple(
        img_left_L,  # type: ignore
        img_left_R,  # type: ignore
        *properties,
    )

    disparity_right = dp.disparity_simple(
        img_right_L,  # type: ignore
        img_right_R,  # type: ignore
        *properties,
    )

    # ToDo: Add disparity cutoff threshold
    # disparity[disparity < 100] = 255

    # calculate the 3D model point cloud
    final_point_cloud = modelbuilder.calculate_point_cloud_final_model(
        disparity_front,
        disparity_back,
        disparity_left,
        disparity_right,
        baseline,
        fov,
        z_scale,
    )

    # pandas data frame for the scatter plot
    points = np.asarray(final_point_cloud.points)
    frm = pd.DataFrame(data=points, columns=["X", "Y", "Z"])

    pc_fig = px.scatter_3d(
        frm,
        x="X",
        y="Z",
        z="Y",
        color="Z",
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    # create figures to show on website
    titles = ["Point Cloud"]
    figures = [
        pc_fig,
    ]

    return titles, figures


layout = [
    dcc.Store(id="memory-final-model", storage_type="session"),
    image_picker_final_model.layout,
    stereo_properties.layout,
    navbar.layout,
    html.Div(
        dcc.Loading(html.Div([], id="graphs-out-model")),
        style={"margin": "0 auto", "width": "50%", "textAlign": "start"},
    ),
]


@app.callback(
    output=[
        dash.Output("graphs-out-model", "children"),
        dash.Output("memory-final-model", "data"),
    ],
    inputs={
        "image_buttons": [
            dash.Input(image_id[0].stem, "n_clicks")
            for image_id in assets.get_asset_images_final_model()
        ],
        "stereo_properties": [dash.Input(prop, "value") for prop in PROPERTIES],
        "resolution": dash.Input("resolution_stereo", "value"),
        "z_scale": dash.Input("z_scale_stereo", "value"),
    },
    state={"memory": dash.State("memory-final-model", "data")},
)
def callback_final_model(
    image_buttons: list,
    stereo_properties: List[int],
    resolution: float,
    z_scale: str,
    memory: dict,
):
    """Callback function for the 3D model calculation."""
    if memory is None:
        memory = {}

    # Get the callback context
    ctx = dash.callback_context
    # Get the id of the input that was changed
    prop_id: str = ctx.triggered[0]["prop_id"]

    # Only image button has .n_clicks property.
    # Thus if its still the same as before, it was only a value change
    is_property = prop_id.replace(".n_clicks", "") == prop_id

    selected_image: Optional[str] = memory.get("selected_image", None)
    image_path: Optional[str] = selected_image

    # no current image selected, but input values changed. -> Do nothing
    if is_property and selected_image is None:
        return None, {}

    elif not is_property:  # Image button was clicked
        # inside the buttons id we stored its asset path, thus remove nclicks
        image_path = prop_id.replace(".n_clicks", "")

    # Load the left and right image
    img_dict = assets.get_asset_image_dict_final_model(image_path)

    titles, figures = update_final_model(
        img_dict, stereo_properties, resolution, _select_scaling(z_scale)
    )

    memory["selected_image"] = image_path
    return graphs.create_graph_card_vertical(titles, figures), memory
