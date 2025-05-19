from dash import html
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H5("Haz clic para aplicar el proceso ETL"), width=12),
            dbc.Col(dbc.Button("Aplicar ETL", id="apply-etl-btn", color="success", class_name="mb-3"), width="auto")
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='etl-output'), width=12)
        ])
    ])