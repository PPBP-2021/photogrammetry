import dash
from dash import dcc
from dash import html

from dashboard import home
from dashboard import image_segmentation
from dashboard import litophane
from dashboard import stereo_point_cloud
from dashboard.instance import app

app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")]
)


@app.callback(dash.Output("page-content", "children"), [dash.Input("url", "pathname")])
def display_page(pathname: str):
    pathname = pathname.lower()

    return {
        "/": home.layout,
        "/home": home.layout,
        "/segmentation": image_segmentation.layout,
        "/litophane": litophane.layout,
        "/stereo_point_cloud": stereo_point_cloud.layout,
    }.get(pathname, "404")
