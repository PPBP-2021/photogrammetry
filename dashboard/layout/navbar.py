import dash
from dash import html
from dash import dcc


import dash_bootstrap_components as dbc


layout = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Image Segmentation",
                                     href="segmentation"),
                dbc.DropdownMenuItem("Litophane", href=""),
                dbc.DropdownMenuItem("Stereo Litophane", href=""),
            ],
            nav=True,
            in_navbar=True,
            label="Modules",
        ),
    ],
    brand="👁👅👁 🗿 Photogrammetry Practical",
    brand_href="/",
    color="primary",
    dark=True,
)
