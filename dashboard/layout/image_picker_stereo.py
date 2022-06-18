import functools

import dash_bootstrap_components as dbc
from dash import html

import dashboard.layout_utils.assets as assets


# ToDo, Create ImagePicker for segmentate images


@functools.lru_cache()
def get_image_cards():
    cards = []
    for image in assets.get_asset_images_stereo():
        card = dbc.Card(
            [
                dbc.Button(
                    dbc.CardImg(
                        src=str(image[0]).replace("dashboard", "."),
                        top=True,
                    ),
                    id=image[0].stem,
                    style={"padding": "0"},
                    color="secondary",
                )
            ]
        )
        cards.append(card)
    return cards


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "25%",
    "padding": "4rem 1rem 2rem",
    "background-color": "#f8f9fa",
    "overflow-y": "scroll",
}


layout = html.Div(
    [
        html.H2("Select Image", className="display-4"),
        html.Hr(),
        html.P("Select any image", className="lead"),
        dbc.Container(get_image_cards()),
    ],
    style=SIDEBAR_STYLE,
)
