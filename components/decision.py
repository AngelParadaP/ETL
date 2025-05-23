from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dcc.Loading([
            html.Div(id='decision-content')
        ], type='circle')
    ])
