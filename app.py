import cv2

import dash
from dash import html
from dash import dcc
import dash_bootstrap_components as dbc

import plotly.express as px
import plotly.graph_objs as go

from modelbuilder import litophane_from_stereo, calculate_disparity, match_keypoints

from dashboard.index import app


if __name__ == "__main__":

    app.run_server(debug=True)
