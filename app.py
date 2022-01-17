import cv2

import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go

from modelbuilder import litophane_from_stereo, calculate_disparity, match_keypoints

app = dash.Dash(__name__, external_stylesheets=[
                dbc.themes.SANDSTONE])


stereo_left_img = cv2.imread("testimages/monke_L.png", 0)
stereo_right_img = cv2.imread("testimages/monke_R.png", 0)

left_points, right_points, img_with_matches = match_keypoints(
    stereo_left_img, stereo_right_img)


app.layout = html.Div([

    dcc.Markdown("""
# StereoVision Litophane
------------------------
"""),

    dbc.CardHeader(
        dbc.Tabs(
            [
                dbc.Tab(label="Matches", tab_id="tab-matches"),
                dbc.Tab(label="Disparity", tab_id="tab-disparity"),
            ],
            id="card-tabs",
            active_tab="tab-matches",)
    ),

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col(html.Label("minDisparity"), width=1),
                dbc.Col(dcc.Slider(id="minDisparity", min=0, max=20, step=1, value=0,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
            dbc.Row([
                dbc.Col(html.Label("numDisparities"), width=1),
                dbc.Col(dcc.Slider(id="numDisparities", min=16, max=10*16, step=16, value=5*16,
                                   tooltip={"placement": "bottom", "always_visible": True})),
            ]),
            dbc.Row([

                dbc.Col(html.Label("window_size"), width=1),
                dbc.Col(dcc.Slider(id="window_size", min=0, max=20, step=1, value=5,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
            dbc.Row([
                dbc.Col(html.Label("disp12MaxDiff"), width=1),
                dbc.Col(dcc.Slider(id="disp12MaxDiff", min=0, max=400, step=1, value=12,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
            dbc.Row([
                dbc.Col(html.Label("uniquenessRatio"), width=1),
                dbc.Col(dcc.Slider(id="uniquenessRatio", min=0, max=60, step=1, value=10,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
            dbc.Row([
                dbc.Col(html.Label("speckleWindowSize"), width=1),
                dbc.Col(dcc.Slider(id="speckleWindowSize", min=0, max=300, step=1, value=50,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
            dbc.Row([
                dbc.Col(html.Label("speckleRange"), width=1),
                dbc.Col(dcc.Slider(id="speckleRange", min=0, max=300, step=1, value=32,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
            dbc.Row([
                dbc.Col(html.Label("preFilterCap"), width=1),
                dbc.Col(dcc.Slider(id="preFilterCap", min=0, max=128, step=1, value=63,
                                   tooltip={"placement": "bottom", "always_visible": True}))
            ]),
        ]),
    ),


    dbc.Card(
        dbc.CardBody([dcc.Loading(
            id="loading-1",
            children=[html.Div([dcc.Graph(
                id="graph-disparity",
                style={"height": "700px"},
                figure=px.imshow(calculate_disparity(left_points,
                                                     right_points,
                                                     stereo_left_img,
                                                     stereo_right_img),
                                 color_continuous_scale="gray"))],
                style={'textAlign': 'center'})
            ],
            type="circle"
        )]))

])


@app.callback(
    dash.dependencies.Output("graph-disparity", "figure"),
    [dash.dependencies.Input("card-tabs", "active_tab"),
     dash.dependencies.Input("minDisparity", "value"),
     dash.dependencies.Input("numDisparities", "value"),
     dash.dependencies.Input("window_size", "value"),
     dash.dependencies.Input("disp12MaxDiff", "value"),
     dash.dependencies.Input("uniquenessRatio", "value"),
     dash.dependencies.Input("speckleWindowSize", "value"),
     dash.dependencies.Input("speckleRange", "value"),
     dash.dependencies.Input("preFilterCap", "value"),
     ]
)
def update_output(active_tab, *args):
    if active_tab == "tab-disparity":
        return px.imshow(
            calculate_disparity(
                left_points, right_points, stereo_left_img, stereo_right_img, *args
            ),
            color_continuous_scale="gray",

        ).update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')
    elif active_tab == "tab-matches":
        return px.imshow(img_with_matches).update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)')


if __name__ == "__main__":
    app.run_server(debug=True)
