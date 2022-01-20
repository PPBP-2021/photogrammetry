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
    "width": "18%",
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
            dcc.Input(name="minDisparity",id="minDisparity",type="number",value=0),
            html.P("numDisparities", className="lead"),
            dcc.Input(name="numDisparities",id="numDisparities",type="number",value=5*16),
            html.P("window_size", className="lead"),
            dcc.Input(name="window_size",id="window_size",type="number",value=5),
            html.P("disp12MaxDiff", className="lead"),
            dcc.Input(name="disp12MaxDiff",id="disp12MaxDiff",type="number",value=12),
            html.P("uniquenessRatio", className="lead"),
            dcc.Input(name="uniquenessRatio",id="uniquenessRatio",type="number",value=10),
            html.P("speckleWindowSize", className="lead"),
            dcc.Input(name="speckleWindowSize",id="speckleWindowSize",type="number",value=50),
            html.P("speckleRange", className="lead"),
            dcc.Input(name="speckleRange",id="speckleRange",type="number",value=32),
            html.P("preFilterCap", className="lead"),
            dcc.Input(name="preFilterCap",id="preFilterCap",type="number",value=63),
            ]
        ),

    ], style=PROPERTY_STYLE
)
