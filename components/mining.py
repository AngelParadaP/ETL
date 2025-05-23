from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dcc.Store(id='mining-models'),  # Para almacenar modelos entrenados
        html.H4("Análisis de Minería de Datos", className="my-4"),

        # Tabs para las diferentes técnicas
        dcc.Tabs([
            # Tab 1: Segmentación de Clientes
            dcc.Tab(label="Segmentación de Clientes", children=[
                html.Div([
                    html.H4("Configuración del Clustering", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Variables para clustering:"),
                            dcc.Dropdown(
                                id='cluster-features',
                                options=[
                                    {'label': 'Lead Time', 'value': 'lead_time'},
                                    {'label': 'Duración de estadía', 'value': 'total_nights'},
                                    {'label': 'ADR (Tarifa diaria)', 'value': 'adr'},
                                    {'label': 'Antigüedad como cliente', 'value': 'previous_bookings_not_canceled'},
                                    {'label': 'Noches de fin de semana', 'value': 'stays_in_weekend_nights'},
                                    {'label': 'Solicitudes especiales', 'value': 'total_of_special_requests'}
                                ],
                                value=['lead_time', 'total_nights', 'adr'],
                                multi=True
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Número de clusters:"),
                            dcc.Slider(
                                id='n-clusters',
                                min=2,
                                max=6,
                                step=1,
                                value=3,
                                marks={i: str(i) for i in range(2, 7)}
                            )
                        ], width=6)
                    ], className="mb-4"),
                    
                    dbc.Button("Ejecutar Clustering", id="run-clustering", color="primary", className="mb-4"),
                    
                    dcc.Loading(
                        id="loading-clustering",
                        children=[
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='cluster-plot'), width=8),
                                dbc.Col(dcc.Graph(id='cluster-boxplot'), width=4)
                            ]),
                            html.H4("Interpretación de Clusters", className="mt-4"),
                            html.Div(id='cluster-insights'),
                            dash_table.DataTable(
                                id='cluster-table',
                                page_size=10,
                                style_table={'overflowX': 'auto'}
                            )
                        ],
                        type="circle"
                    )
                ])
            ]),
            

            # Tab 2: Predicción de Ocupación 
           dcc.Tab(label="Predicción de Ocupación", children=[
                html.Div([
                    html.H4("Configuración del Modelo de Ocupación", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Variables predictoras:"),
                            dcc.Dropdown(   
                                id='prediction-features',
                                options=[
                                    {'label': 'Año de llegada', 'value': 'arrival_date_year'},
                                    {'label': 'Mes de llegada', 'value': 'arrival_date_month'},
                                    {'label': 'Tipo de Hotel', 'value': 'hotel'},
                                    {'label': 'ADR (Tarifa Diaria Promedio)', 'value': 'adr'},
                                    {'label': 'Lead Time', 'value': 'lead_time'},
                                    {'label': 'Noches en fin de semana', 'value': 'stays_in_weekend_nights'},
                                    {'label': 'Noches entre semana', 'value': 'stays_in_week_nights'},
                                    {'label': 'Solicitudes especiales', 'value': 'total_of_special_requests'}
                                ],
                                value=['arrival_date_year', 'arrival_date_month', 'hotel', 'adr'],
                                multi=True
                            )
                        ], width=8),
                        dbc.Col([
                            html.Label("Tipo de modelo:"),
                            dcc.Dropdown(
                                id='model-type',
                                options=[
                                    {'label': 'Random Forest', 'value': 'random_forest'},
                                    {'label': 'Regresión Lineal', 'value': 'linear'}
                                ],
                                value='random_forest'
                            )
                        ], width=4)
                    ], className="mb-4"),
                    
                    dbc.Button("Entrenar Modelo de Ocupación", id="train-model", color="primary", className="mb-4"),
                    
                    dcc.Loading(
                        id="loading-model",
                        children=[
                            html.Div(id='model-performance'),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='feature-importance'), width=6),
                                dbc.Col(dcc.Graph(id='error-analysis-plot'), width=6)
                            ]),
                            html.H4("Notas del Modelo", className="mt-4"),
                            html.Div(id='model-rules'),
                            html.H4("Predecir Ocupación para Nuevas Fechas", className="mt-4"),
                            html.Div(id='prediction-interface'),
                            html.Div(id='prediction-result', className='mt-4')
                        ],
                        type="circle"
                    )
                ])
            ]),
            
            # Tab 3: Análisis de Ingresos
            dcc.Tab(label="Optimización de Ingresos", children=[
                html.Div([
                    html.H4("Análisis de Precios e Ingresos", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Variable objetivo:"),
                            dcc.Dropdown(
                                id='revenue-target',
                                options=[
                                    {'label': 'ADR (Tarifa diaria)', 'value': 'adr'},
                                    {'label': 'Ingreso total por reserva', 'value': 'total_revenue'}
                                ],
                                value='adr'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Variables predictoras:"),
                            dcc.Dropdown(
                                id='revenue-features',
                                options=[
                                    {'label': 'Lead Time', 'value': 'lead_time'},
                                    {'label': 'Temporada', 'value': 'arrival_date_month'},
                                    {'label': 'Duración estadía', 'value': 'total_nights'},
                                    {'label': 'Tipo de habitación', 'value': 'reserved_room_type'},
                                    {'label': 'Tipo de cliente', 'value': 'customer_type'},
                                    {'label': 'Segmento de mercado', 'value': 'market_segment'}
                                ],
                                value=['lead_time', 'arrival_date_month', 'total_nights'],
                                multi=True
                            )
                        ], width=8)
                    ], className="mb-4"),
                    
                    dbc.Button("Analizar Ingresos", id="analyze-revenue", color="primary", className="mb-4"),
                    
                    dcc.Loading(
                        id="loading-revenue",
                        children=[
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='revenue-regression'), width=6),
                                dbc.Col(dcc.Graph(id='revenue-heatmap'), width=6)
                            ]),
                            html.Div(id='revenue-insights'),
                            html.H4("Recomendaciones de Precios", className="mt-4"),
                            html.Div(id='pricing-recommendations')
                        ],
                        type="circle"
                    )
                ])
            ])
        ]),
                
        html.Div(className="my-4"),
        html.Div(className="my-4"),
    ], fluid=True)