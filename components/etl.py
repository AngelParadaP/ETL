from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dcc.Download(id="download-data"),
        dcc.Store(id='etl-data'),
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
        dbc.Row([
            dbc.Col(dbc.Button("Exportar a PostgreSQL", id="toggle-export-form", color="primary", class_name="mb-3", disabled=True), width="auto")
        ]),
        dbc.Collapse(
            id="export-form-collapse",
            is_open=False,
            children=dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Input(id="db-hosteo", placeholder="Host", type="text"), width=4),
                        dbc.Col(dbc.Input(id="db-puerto", placeholder="Puerto", type="number"), width=2),
                        dbc.Col(dbc.Input(id="db-nombre", placeholder="Nombre de la base de datos", type="text"), width=4),
                    ], class_name="mb-2"),
                    dbc.Row([
                        dbc.Col(dbc.Input(id="db-usuario", placeholder="Usuario", type="text"), width=4),
                        dbc.Col(dbc.Input(id="db-contra", placeholder="Contraseña", type="password"), width=4),
                        dbc.Col(dbc.Input(id="db-tabla", placeholder="Nombre de la tabla", type="text"), width=4),
                    ], class_name="mb-2"),
                    dbc.Button("Exportar a PostgreSQL", id="export-to-db-btn", color="warning", class_name="mt-2")
                ])
            ])
        ),
        dbc.Row([
            dbc.Col(html.Div(id="export-status"), width=12)
        ])
    ])
