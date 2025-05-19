from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
from callbacks import register_callbacks

# Importar componentes
from components.carga import layout as carga_layout
from components.etl import layout as etl_layout


# App
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Sistema Hotelero"

# Layout 
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(dcc.Store(id='store-data')),
        dbc.Col(dcc.Store(id='etl-data')),
        dbc.Col(dbc.Alert("Sistema Hotelero - Almacenes de Datos", color="primary", className="text-center fw-bold fs-3"),width=12)
    ]),

    dbc.Tabs([
        dbc.Tab(carga_layout(), label='Carga de Datos'),
        dbc.Tab(etl_layout(), label='ETL')
    ])
], fluid=True)

# Callbacks
register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)
