from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dcc.Download(id="download-data"),  # Componente esencial para descargas
        dcc.Store(id='etl-data'),  # Asegúrate de que existe
        
        dbc.Row([
            dbc.Col(html.H5("Haz clic para aplicar el proceso ETL"), width=12),
            dbc.Col(dbc.Button("Aplicar ETL", id="apply-etl-btn", color="success", class_name="mb-3"), width="auto")
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='etl-output'), width=12)
        ]),
        dbc.Row([
            dbc.Col(dbc.Button("Descargar CSV", id="download-csv-btn", color="primary", class_name="mb-3"), width="auto"),
            dbc.Col(dbc.Button("Descargar JSON", id="download-json-btn", color="primary", class_name="mb-3"), width="auto"),  
            dbc.Col(dbc.Button("Descargar Excel", id="download-excel-btn", color="primary", class_name="mb-3"), width="auto"),
        ]),
    ])