import dash
import dash_bootstrap_components as dbc


external_stylesheets = [dbc.themes.COSMO, ]
app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets,
    suppress_callback_exceptions=True
)

app_title = "Photogrammetry Practical"
app.title = app_title
