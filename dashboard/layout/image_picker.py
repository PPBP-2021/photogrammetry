from pathlib import Path
import functools
import os
import json
import sys

import dash
from dash import html
from dash import dcc

import dash_bootstrap_components as dbc
from importlib_metadata import pathlib



@functools.lru_cache()
def get_asset_images():
    asset_path = Path("./dashboard/assets")
    configs = [p for p in asset_path.iterdir() if p.suffix == ".json"]
    returns = []
    for config in configs:
        with open(config) as f:
            config = json.load(f)
            returns.append(
                (asset_path/config["left_image"],
                asset_path/config["right_image"],
                config)
            )
            
    return returns


@functools.lru_cache()
def get_image_cards():
    cards = []
    for image in get_asset_images():
        card = dbc.Card(
            [
                dbc.Button(
                    dbc.CardImg(
                        src=str(image[0]).replace("dashboard", "."),
                        top=True,
                    ), id=image[0].stem, style={"padding": "0"}, color="secondary")

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
            "Select any image", className="lead"
        ),
        dbc.Container(
            get_image_cards()
        ),

    ], style=SIDEBAR_STYLE
)
