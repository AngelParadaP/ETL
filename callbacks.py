from dash import Input, Output, State, html, dash_table
import pandas as pd
from ETL import DateProcessor, CleanProcessor
import base64
import io

# Función para leer y mostrar el archivo
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    # Procesar el archivo según su tipo
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            raise ValueError("Formato no soportado")
    except Exception as e:
        return html.Div([
            'Error al procesar el archivo: {}'.format(e)
        ])

    # Mostrar el preview del archivo
    preview = html.Div([
        html.H5(f"Archivo cargado: {filename}"),
        html.P(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}"),
        dash_table.DataTable(
            data=df.head(10).to_dict('records'),
            columns=[{"name": i, "id": i} for i in df.columns],
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "minWidth": "100px"},
            page_size=10,
        )
    ])
    return df, preview

# Callback para cargar el archivo
def register_callbacks(app):
    @app.callback(
        Output('output-data-upload', 'children'),
        Output('store-data', 'data'),
        Input('upload-data', 'contents'),
        State('upload-data', 'filename'),
    )
    def cargar_archivo(contents, filename):
        if contents:
            df, layout = parse_contents(contents, filename)
            if df is not None:
                # Si se cargó correctamente, muestra tabla y guarda los datos
                return layout, df.to_json(date_format='iso', orient='split')
        return None, None
    
# Callback para aplicar el proceso ETL
    @app.callback(
        Output('etl-output', 'children'),
        Output('etl-data', 'data'),
        Input('apply-etl-btn', 'n_clicks'),
        State('store-data', 'data'),
        prevent_initial_call=True
    )
    def aplicar_etl(n_clicks, json_data):
        if not json_data:
            return html.Div(["No hay datos cargados."]), None

        df = pd.read_json(json_data, orient='split')
        df_etl = df.copy()

        # Aplicar los procesadores
        processors = [DateProcessor(), CleanProcessor()]
        for processor in processors:
            df_etl = processor.process(df_etl)

        # Visualizar antes y después del ETL
        return html.Div([
            html.H4("Antes del ETL"),
            dash_table.DataTable(
                data=df.head(5).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'}
            ),
            html.H4("Después del ETL"),
            dash_table.DataTable(
                data=df_etl.head(5).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'}
            )
        ]), df_etl.to_json(date_format='iso', orient='split')