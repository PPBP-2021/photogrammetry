import dash
from dash import html
from dash import dcc


import dash_bootstrap_components as dbc


layout = dbc.NavbarSimple(
    children=[
        dbc.Button("Sidebar", outline=True, color="secondary",
                   className="mr-1", id="btn_sidebar"),
        dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Brand",
    brand_href="#",
    color="dark",
    dark=True,
    fluid=True,
)
