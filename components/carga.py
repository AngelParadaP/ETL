from dash import dcc, html
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        # Sección de carga de archivos
        dbc.Row([
            dbc.Col(html.H4("Carga de Datos", className="mb-4"), width=12),
            dbc.Col(html.H5("Cargar desde Archivo (CSV, JSON, Excel):"), width=12)
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
            dbc.Col(html.H5("Cargar desde PostgreSQL:", className="mt-4"), width=12),
            dbc.Col(
                dbc.Form([
                    dbc.Row([
                        # Inputs con tamaño flexible
                        dbc.Col(
                            dbc.Input(
                                id="db-host", 
                                placeholder="Host", 
                                type="text",
                                style={"minWidth": "200px"}
                            ),
                            className="mb-2",
                            style={"minWidth": "200px", "flexGrow": 1}
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="db-port", 
                                placeholder="Port", 
                                type="number",
                                style={"minWidth": "100px"}
                            ),
                            className="mb-2",
                            style={"minWidth": "100px", "flexGrow": 0.5}
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="db-name", 
                                placeholder="Database", 
                                type="text",
                                style={"minWidth": "150px"}
                            ),
                            className="mb-2",
                            style={"minWidth": "150px", "flexGrow": 1}
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="db-user", 
                                placeholder="User", 
                                type="text",
                                style={"minWidth": "120px"}
                            ),
                            className="mb-2",
                            style={"minWidth": "120px", "flexGrow": 0.8}
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="db-password", 
                                placeholder="Password", 
                                type="password",
                                style={"minWidth": "130px"}
                            ),
                            className="mb-2",
                            style={"minWidth": "130px", "flexGrow": 0.8}
                        ),
                        dbc.Col(
                            dbc.Input(
                                id="db-table", 
                                placeholder="Table", 
                                type="text",
                                style={"minWidth": "150px"}
                            ),
                            className="mb-2",
                            style={"minWidth": "150px", "flexGrow": 1}
                        ),
                    ], 
                    className="g-2",  # Espacio entre elementos
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "alignItems": "center",
                        "justifyContent": "start"
                    }),
                    
                    dbc.Row([
                        dbc.Col(
                            dbc.Button("Conectar y Cargar Datos", 
                                    id="db-connect-btn", 
                                    color="primary",
                                    className="mt-3"),
                            width="auto"
                        )
                    ])
                ]), 
                width=12
            )
        ]),
        dbc.Container([
            dcc.Store(id='last-data-source', data=None),
            dcc.Store(id='force-refresh', data=0),  # Nuevo componente para forzar actualizaciones
            
            # Sección de resultados
            dbc.Row([
                dbc.Col(
                    html.Div(id='dynamic-output-container', children=[
                        html.Div(id='output-data-upload', style={'display': 'none'}),
                        html.Div(id='output-db-connection', style={'display': 'none'})
                    ]), 
                    width=12
                )
            ])
        ])
    ], fluid=True)