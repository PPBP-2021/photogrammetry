import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

PROPERTY_STYLE = {
    "position": "fixed",
    "top": 0,
    "bottom": 0,
    "right": 0,
    "width": "25%",
    "padding": "4rem 1rem 2rem",
    "background-color": "#f8f9fa",
    "overflow-y": "scroll",
}


layout = html.Div(
    [
        html.H2("Properties", className="display-10"),
        html.Hr(),
        dbc.Container(
            [
                html.P(
                    html.Abbr(
                        "Grayscale Threshold",
                        title="Threshold for grayscale image. The value is between 0 and 255.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="grayscaleThreshold",
                    min=0,
                    step=1,
                    max=255,
                    value=240,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
            ]
        ),
    ],
    style=PROPERTY_STYLE,
)
