import dash
from dash import html
from dash import dcc


import dash_bootstrap_components as dbc

from dashboard import home
from dashboard import image_segmentation
from dashboard import litophane_from_stereo
from dashboard.instance import app


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


@app.callback(dash.Output("page-content", "children"),
              [dash.Input("url", "pathname")])
def display_page(pathname):
    if pathname.lower() in ("/", "/home"):
        return home.layout
    elif pathname == "/segmentation":
        return image_segmentation.layout
    elif pathname == "/litophane_from_stereo":
        return litophane_from_stereo.layout
    else:
        return "404"
