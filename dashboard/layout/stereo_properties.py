import glob
import functools
import os

import dash
from dash import html
from dash import dcc


import dash_bootstrap_components as dbc
from matplotlib.pyplot import text
PROPERTY_STYLE = {
    "position": "fixed",
    "top": 0,
    "right": 0,
    "bottom": 0,
    "width": "25%",
    "padding": "4rem 1rem 2rem",
    "background-color": "#f8f9fa",
}


layout = html.Div(
    [

        html.H2("Properties", className="display-10"),
        html.Hr(),
        dbc.Container(
            [
            html.P("minDisparities", className="lead"),
            dcc.Slider(id="minDisparity",min=0,step=1,max=500,value=0,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("numDisparities", className="lead"),
            dcc.Slider(id="numDisparities",min=1,step=16,max=10*16,value=5*16,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("window_size", className="lead"),
            dcc.Slider(id="window_size",min=1,step=2,max=31,value=5,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("disp12MaxDiff", className="lead"),
            dcc.Slider(id="disp12MaxDiff",min=-1,step=1,max=100,value=12,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("uniquenessRatio", className="lead"),
            dcc.Slider(id="uniquenessRatio",min=1,step=1,max=100,value=10,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("speckleWindowSize", className="lead"),
            dcc.Slider(id="speckleWindowSize",min=0,step=50,max=200,value=50,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("speckleRange", className="lead"),
            dcc.Slider(id="speckleRange",min=-1,step=1,max=5,value=5,tooltip={"placement": "bottom", "always_visible": True}),
            html.P("preFilterCap", className="lead"),
            dcc.Slider(id="preFilterCap",min=0,step=1,max=126,value=63,tooltip={"placement": "bottom", "always_visible": True}),
            ]
        ),

    ], style=PROPERTY_STYLE
)
