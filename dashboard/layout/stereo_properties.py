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
                        "minDisparities",
                        title="Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="minDisparity",
                    min=-128,
                    step=16,
                    max=128,
                    value=-64,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "numDisparities",
                        title="Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="numDisparities",
                    min=16,
                    step=16,
                    max=12 * 16,
                    value=12 * 16,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "block_size",
                        title="Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="block_size",
                    min=1,
                    step=2,
                    max=15,
                    value=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "disp12MaxDiff",
                        title="Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="disp12MaxDiff",
                    min=-1,
                    step=1,
                    max=50,
                    value=4,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "uniquenessRatio",
                        title="Margin in percentage by which the best (minimum) computed cost function value should 'win' the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="uniquenessRatio",
                    min=1,
                    step=1,
                    max=25,
                    value=5,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "speckleWindowSize",
                        title="Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="speckleWindowSize",
                    min=0,
                    step=50,
                    max=200,
                    value=200,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "speckleRange",
                        title="Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="speckleRange",
                    min=-1,
                    step=1,
                    max=5,
                    value=2,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "Resolution",
                        title="Adjust resolution of image to save computation time",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="resolution_stereo",
                    min=0.1,
                    step=0.1,
                    max=1,
                    value=1,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "Disparity Treshold",
                        title="Adjust disparity cutoff to cutoff far away objects",
                    ),
                    className="lead",
                ),
                dcc.Slider(
                    id="treshold_stereo",
                    min=0,
                    step=1,
                    max=254,
                    value=200,
                    tooltip={"placement": "bottom", "always_visible": True},
                    marks=None,
                ),
                html.P(
                    html.Abbr(
                        "3D z-scale",
                        title="Choose the method to scale the depth values (z) of the point_cloud",
                    ),
                    className="lead",
                ),
                dcc.RadioItems(
                    id="z_scale_stereo",
                    options=[
                        {"label": "No Scale", "value": "no"},
                        {"label": "log Scale", "value": "log"},
                        {"label": "Quadratic Scale", "value": "quadratic"},
                    ],
                    value="no",
                    labelStyle={"display": "block"},
                ),
            ]
        ),
    ],
    style=PROPERTY_STYLE,
)
