from typing import List, Optional, Tuple, cast

import cv2
import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html
from imageprocessing import segmentate_grayscale
from modelbuilder import litophane_from_image

import dashboard.layout_utils.assets as assets
import dashboard.layout_utils.graphs as graphs
from dashboard.instance import app
from dashboard.layout import image_picker, litophane_properties, navbar

CURRENT_SEG = None
CURRENT_PATH = None


def convert_stl_mesh_to_figure(stl_mesh):
    def stl2mesh3d(stl_mesh):
        # stl_mesh is read by nympy-stl from a stl file; it is  an array of faces/triangles (i.e. three 3d points)
        # this function extracts the unique vertices and the lists I, J, K to define a Plotly mesh3d
        p, q, r = stl_mesh.vectors.shape  # (p, 3, 3)
        # the array stl_mesh.vectors.reshape(p*q, r) can contain multiple copies of the same vertex;
        # extract unique vertices from all mesh triangles
        vertices, ixr = np.unique(stl_mesh.vectors.reshape(
            p*q, r), return_inverse=True, axis=0)
        I = np.take(ixr, [3*k for k in range(p)])
        J = np.take(ixr, [3*k+1 for k in range(p)])
        K = np.take(ixr, [3*k+2 for k in range(p)])
        return vertices, I, J, K

    vertices, I, J, K = stl2mesh3d(stl_mesh)
    x, y, z = vertices.T

    colorscale = [[0, '#e5dee5'], [1, '#e5dee5']]
    mesh3D = go.Mesh3d(x=x,
                       y=y,
                       z=z,
                       i=I,
                       j=J,
                       k=K,
                       flatshading=True,
                       colorscale=colorscale,
                       intensity=z,
                       name="Litophane",
                       showscale=False)

    layout = go.Layout(paper_bgcolor='rgb(1,1,1)',
                       title_text="Liptophane", title_x=0.5,
                       font_color='white',
                       scene_camera=dict(eye=dict(x=1.25, y=-1.25, z=1)),
                       scene_xaxis_visible=False,
                       scene_yaxis_visible=False,
                       scene_zaxis_visible=False,
                       scene=dict(
                           aspectmode='data'
                       ))
    fig = go.Figure(data=[mesh3D], layout=layout)
    fig.data[0].update(lighting=dict(ambient=0.18,
                                     diffuse=1,
                                     fresnel=.1,
                                     specular=1,
                                     roughness=.1,
                                     facenormalsepsilon=0))
    return fig


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


def select_scaling(function):
    if function == "log":
        return lambda z: np.log(z)
    elif function == "quadratic":
        return lambda z: np.square(z)
    elif function == "no":
        return lambda z: z


def update_grayscale(treshold, asset_images=None, image_path=None):
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
        dash.Input("z_scale", "value")
    ]
)
def select_image(*inputs):
    global CURRENT_SEG

    asset_images = assets.get_asset_images()

    # update all our property values
    current_treshold = inputs[-2]
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
        update_grayscale(current_treshold, asset_images, image_path)

    # The input that triggered this callback was a property change
    else:
        prop = prop_id.replace(".value", "")
        # The input property was the threshold for our grayscale
        if prop == "gray_treshold":
            update_grayscale(current_treshold)

    seg_fig = _create_segmentation_fig(CURRENT_SEG)
    lito_mesh = litophane_from_image(
        CURRENT_SEG, z_scale=select_scaling(z_scale))
    lito_fig = convert_stl_mesh_to_figure(lito_mesh)
    titles, figures = ["Segmentated", "3D Litophane"], [seg_fig, lito_fig]

    return graphs.create_graph_card_vertical(titles, figures)
