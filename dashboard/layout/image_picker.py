import glob
import functools
import os

import dash
from dash import html
from dash import dcc


import dash_bootstrap_components as dbc


def get_asset_images(fullpath=False):
    images = glob.glob("./**/*.png", recursive=True) + \
        glob.glob("./**/*.jpg", recursive=True)
    if not fullpath:
        images = [path[path.find("assets"):] for path in images]
    # remove stuff that isnt png or jpg
    images = [path for path in images if len(path) > 3]
    return images


@functools.lru_cache()
def get_image_cards():
    cards = []
    for image in get_asset_images():
        card = dbc.Card(
            [
                dbc.Button(
                    dbc.CardImg(
                        src=image,
                        top=True,
                    ), id=os.path.splitext(image)[0], style={"padding": "0"}, color="secondary")

            ]

        )
        cards.append(card)
    return cards


SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "15%",
    "padding": "4rem 1rem 2rem",
    "background-color": "#f8f9fa",
    "overflow": "scroll"
}


layout = html.Div(
    [

        html.H2("Select Image", className="display-4"),
        html.Hr(),
        html.P(
            "Select any image that you want to seperate from its background.", className="lead"
        ),
        dbc.Container(
            get_image_cards()
        ),

    ], style=SIDEBAR_STYLE
)
