from dash import dcc, html
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H5("Cargar Archivo (CSV, JSON o Excel):"), width=12)
        ]),
        dbc.Row([
            dbc.Col(dcc.Upload(
                id='upload-data',
                children=html.Div(['Arrastra o haz clic para subir'],
                                  style={'textAlign': 'center'}),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '2px',
                    'borderStyle': 'dashed',
                    'borderRadius': '10px',
                    'backgroundColor': '#fdfdfd',
                    'marginBottom': '20px'
                },
                multiple=False
            ), width=12),
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='output-data-upload'), width=12)
        ])
    ])  