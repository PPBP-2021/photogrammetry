import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from dashboard import (home, image_segmentation, litophane,
                       litophane_from_stereo)
from dashboard.instance import app

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


@app.callback(dash.Output("page-content", "children"),
              [dash.Input("url", "pathname")])
def display_page(pathname: str):
    pathname = pathname.lower()

    return {
        "/": home.layout,
        "/home": home.layout,
        "/segmentation": image_segmentation.layout,
        "/litophane": litophane.layout,
        "/litophane_from_stereo": litophane_from_stereo.layout
    }.get(pathname, "404")
