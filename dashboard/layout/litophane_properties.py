import dash_bootstrap_components as dbc
from dash import dcc, html

PROPERTY_STYLE = {
    "position": "fixed",
    "top": 0,
    "bottom": 0,
    "right": 0,
    "width": "25%",
    "padding": "4rem 1rem 2rem",
    "background-color": "#f8f9fa",
    "overflow-y": "scroll"
}


layout = html.Div(
    [
        html.H2("Properties", className="display-10"),
        html.Hr(),
        dbc.Container(
            [
                html.P(
                    html.Abbr(
                        "grayscale treshold", title="Grayscaled pixels with a higher value than this will be turned completely black."
                    ), className="lead"
                ),
                dcc.Slider(id="gray_treshold", min=50, step=1, max=255, value=240, tooltip={
                           "placement": "bottom", "always_visible": True}),

                html.P(
                    html.Abbr(
                        "resolution", title="Adjust resolution for the litophane creation to save computation time"
                    ), className="lead"
                ),
                dcc.Slider(id="resolution", min=0.1, step=0.1, max=1, value=0.5, tooltip={
                           "placement": "bottom", "always_visible": True}),

                html.P(
                    html.Abbr(
                        "3D z-scale", title="Chose mehtod to scale the depth values (z) of the litophane"
                    ), className="lead"
                ),
                dcc.RadioItems(
                    id="z_scale",
                    options=[
                        {"label": "no scale", "value": "no"},
                        {"label": "log", "value": "log"},
                        {"label": "quadratic", "value": "quadratic"}
                    ],
                    value="no",
                    labelStyle={"display": "block"}
                )
            ]
        ),

    ], style=PROPERTY_STYLE
)
