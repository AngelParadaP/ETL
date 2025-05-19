from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

def layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H5("Análisis Exploratorio de Datos"), width=12, className="mb-4")
        ]),
        
        # Sección de Estadísticas y Gráficos
        dbc.Row([
            # Columna de Estadísticas
            dbc.Col([
                html.H6("Estadísticas Descriptivas", className="mb-3"),
                dcc.Dropdown(
                    id='eda-stat-dropdown',
                    options=[
                        {'label': 'Tiempo de Espera', 'value': 'lead_time'},
                        {'label': 'Precio Diario', 'value': 'adr'},
                        {'label': 'Noches Totales', 'value': 'total_nights'}
                    ],
                    value='lead_time',
                    className="mb-3"
                ),
                dbc.Card([
                    dbc.CardBody([
                        html.Div(id='eda-statistics-content', className="small")
                    ])
                ])
            ], width=4),
            
            # Columna de Gráficos
            dbc.Col([
                dbc.Row([
                    # Histograma
                    dbc.Col([
                        html.H6("Distribución de Variables", className="mb-3"),
                        dcc.Dropdown(
                            id='eda-hist-dropdown',
                            options=[
                                {'label': 'Tiempo de Espera', 'value': 'lead_time'},
                                {'label': 'Precio Diario', 'value': 'adr'},
                                {'label': 'Noches Totales', 'value': 'total_nights'}
                            ],
                            value='lead_time',
                            className="mb-3"
                        ),
                        dcc.Graph(id='eda-histogram-plot')
                    ], width=6),
                    
                    # Boxplot
                    dbc.Col([
                        html.H6("Análisis de Outliers", className="mb-3"),
                        dcc.Dropdown(
                            id='eda-boxplot-dropdown',
                            options=[
                                {'label': 'Precio Diario', 'value': 'adr'},
                                {'label': 'Tiempo de Espera', 'value': 'lead_time'},
                                {'label': 'Días en Lista de Espera', 'value': 'days_in_waiting_list'}
                            ],
                            value='adr',
                            className="mb-3"
                        ),
                        dcc.Graph(id='eda-boxplot-plot')
                    ], width=6)
                ]),
                
                # Serie Temporal
                dbc.Row([
                    dbc.Col([
                        html.H6("Tendencia Temporal", className="mt-4"),
                        dcc.Graph(id='eda-time-series-plot')
                    ], width=12)
                ])
            ], width=8)
        ])
    ], fluid=True)