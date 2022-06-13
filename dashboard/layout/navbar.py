import dash
import dash_bootstrap_components as dbc
from dash import dcc
from dash import html


layout = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("Image Segmentation", href="segmentation"),
                dbc.DropdownMenuItem("Litophane", href="litophane"),
                dbc.DropdownMenuItem("Stereo Point Cloud", href="stereo_point_cloud"),
                dbc.DropdownMenuItem("3D Model", href="final_model"),
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
