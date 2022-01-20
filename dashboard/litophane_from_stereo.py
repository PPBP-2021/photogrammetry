from curses import window
import os

import dash
from dash import html
from dash import dcc
from matplotlib.pyplot import figure, title

import plotly.express as px

import dash_bootstrap_components as dbc

from dashboard.layout import image_picker
from dashboard.layout import navbar
from dashboard.layout import stereo_properties
from dashboard.instance import app

import modelbuilder.litophane

import cv2


PROPERTIES = ["minDisparity", "numDisparities", "window_size", "disp12MaxDiff",
              "uniquenessRatio", "speckleWindowSize", "speckleRange", "preFilterCap"]

current_left_image = None
current_right_image = None
property_values = [0,5*16,5,12,10,50,32,63]


def create_graph_card_horizontal(titles, graphs):
    content = []

    for title, graph in zip(titles, graphs):
        content.append(
            dbc.Card(
                [
                    dbc.CardHeader(title),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=graph,
                            config={"displayModeBar": False}
                        )
                    )
                ], class_name="w-30 mb-3"
            ),
        )

    return dbc.Container(content)


def calculate_current_disparity():
    left_points, right_points, _ = modelbuilder.litophane.match_keypoints(
        current_left_image, current_right_image)

    disparity = modelbuilder.litophane.calculate_disparity(left_points, right_points,
                                                           current_left_image, current_right_image, *property_values)

    titles = ["Disparity Map"]
    figures = [px.imshow(disparity, color_continuous_scale="gray").update_layout(
        margin=dict(
            b=0,  # bottom margin 40px
            l=0,  # left margin 40px
            r=0,  # right margin 20px
            t=0,  # top margin 20px
        )
    )]

    return titles, figures


layout = [
    image_picker.layout,
    stereo_properties.layout,
    navbar.layout,
    html.Div([

    ], id="graphs-out-stereo")
]


@app.callback(dash.Output("graphs-out-stereo", "children"),
              [dash.Input(image_id[0].stem, "n_clicks")
               for image_id in image_picker.get_asset_images()] + [dash.Input(prop, "value")
               for prop in PROPERTIES],
              )
def select_image(*inputs):

    asset_images = image_picker.get_asset_images()
    global current_left_image
    global current_right_image
    global property_values
    property_values = inputs[len(asset_images):]

    prop_id = dash.callback_context.triggered[0]["prop_id"]
    try_replace_n_clicks = prop_id.replace(
        ".n_clicks", "")
    
    print(prop_id)

    if prop_id == try_replace_n_clicks and current_left_image is None:
        print("case1")
        return
    elif prop_id != try_replace_n_clicks:
        print("case2")
        image_path = dash.callback_context.triggered[0]["prop_id"].replace(
            ".n_clicks", "")

        file_path = ""
        for image in asset_images:
            if image_path in str(image[0]):
                file_path_left = str(image[0])
                file_path_right = str(image[1])
                break

        current_left_image = cv2.cvtColor(
            cv2.imread(file_path_left), cv2.COLOR_BGR2GRAY)
        current_right_image = cv2.cvtColor(
            cv2.imread(file_path_right), cv2.COLOR_BGR2GRAY)

    titles, figures = calculate_current_disparity()

    return create_graph_card_horizontal(titles, figures)

