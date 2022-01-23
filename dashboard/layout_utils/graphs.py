from typing import List

import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from dash import dcc


def create_graph_card_vertical(titles: List[str], graphs: List[go.Figure]) -> dbc.Container:
    """Create a List of graphs in Cards, places vertically.

    Parameters
    ----------
    titles : List[str]
        The Titles for the cards in order of the corresponsing graphs/figures.
    graphs : List[go.Figure]
        The Figures for the cards in order of the corresponding titles.

    Returns
    -------
    dbc.Container
        A dbc.Conatiner with all the Cards in vertical orientation.
    """
    content = []

    for title, graph in zip(titles, graphs):
        content.append(
            dbc.Card(
                [
                    dbc.CardHeader(title),
                    dbc.CardBody(
                        dcc.Graph(
                            figure=graph,
                            config={"displayModeBar": False}
                        )
                    )
                ], class_name="w-30 mb-3"
            ),
        )

    return dbc.Container(content)
