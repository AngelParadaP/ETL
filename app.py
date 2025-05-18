import base64
import io
import pandas as pd
from dash import Dash, dcc, html, Output, Input, State, dash_table
from ETL import DataProcessor, CleanProcessor


app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1("Proyecto Final - Almacén de Datos", style={'textAlign': 'center'}),
   
   html.H3("Sube tu archivo (CSV, JSON o Excel)"),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra un archivo aquí o haz clic para subir.']),
        className='dash-upload',
        multiple=False
    ),
    html.Div(id='output-data-upload')
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['Formato de archivo no soportado.'])
    except Exception as e:
        return html.Div([f'Ocurrió un error al procesar el archivo: {e}'])

    return html.Div([
        html.H5(f"Archivo cargado: {filename}"),
        html.P(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}"),
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{'name': col, 'id': col} for col in df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
        )
    ])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        return parse_contents(contents, filename)

if __name__ == '__main__':
    app.run(debug=True)