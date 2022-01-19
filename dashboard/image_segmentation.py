import os

import dash
from dash import html
from dash import dcc

import plotly.express as px

import dash_bootstrap_components as dbc

from dashboard.layout import image_picker
from dashboard.layout import navbar
from dashboard.instance import app


from imageprocessing import segmentate_grayscale

import cv2


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


layout = [
    image_picker.layout,
    navbar.layout,
    html.Div([

    ], id="graphs-out")
]


@app.callback(dash.Output("graphs-out", "children"),
              [dash.Input(os.path.splitext(image_id)[0], "n_clicks") for image_id in image_picker.get_asset_images()])
def select_image(*image_path):

    image_path = dash.callback_context.triggered[0]["prop_id"].replace(
        ".n_clicks", "")

    file_path = ""
    for image in image_picker.get_asset_images(True):
        if image_path in image:
            file_path = image
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
            )
        ),

        px.imshow(
            cv2.cvtColor(
                segmentate_grayscale(file_path, 240),
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
    ]

    return create_graph_card_horizontal(titles, figures)
