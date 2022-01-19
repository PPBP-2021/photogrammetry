import dash
from dash import html
from dash import dcc


import dash_bootstrap_components as dbc

from dashboard import home

external_stylesheets = [dbc.themes.SANDSTONE, ]
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=external_stylesheets,
)

app_title = "Photogrammetry Practical"
app.title = app_title


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])


@app.callback(dash.Output("page-content", "children"),
              [dash.Input("url", "pathname")])
def display_page(pathname):
    if pathname == '/':
        return home.layout
    elif pathname == '/segmentation':
        return "404"
    else:
        return "404"
