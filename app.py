from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# Components
from components.carga import layout as carga_layout
from components.etl import layout as etl_layout
from components.mining import layout as mining_layout
from components.decision import layout as decision_layout

# App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = dbc.Container([
    dcc.Store(id='stored-data'), # Store para guardar DataFrame procesado
    html.H1("Sistema Hotelero - Análisis de Datos", className="mb-4"),
    dcc.Tabs(id='tabs', value='carga', children=[
        dcc.Tab(label='Carga de Datos', value='carga'),
        dcc.Tab(label='ETL', value='etl'),
        dcc.Tab(label='Minería de Datos', value='mining'),
        dcc.Tab(label='Decisión', value='decision')
    ]),
    html.Div(id='tabs-content'),
], fluid=True, className="p-4", style={'backgroundColor': '#f8f9fa'})

# Callbacks
@callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_tab(tab_name):
    if tab_name == 'carga':
        return carga_layout
    elif tab_name == 'etl':
        return etl_layout
    elif tab_name == 'mining':
        return mining_layout
    elif tab_name == 'decision':
        return decision_layout
    else:
        return html.Div("Seleccione una pestaña válida.")
    
if __name__ == '__main__':
    app.run_server(debug=True)  
# -*- coding: utf-8 -*-