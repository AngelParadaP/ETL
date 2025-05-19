from dash import Input, Output, State, html, dash_table, dcc, no_update
import pandas as pd
from ETL import DateProcessor, CleanProcessor, PredictiveFeaturesProcessor, SeasonalAnalysisProcessor
import base64
import io
import plotly.express as px
from ETL import DatabaseManager
import dash

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
        processors = [DateProcessor(), CleanProcessor(), PredictiveFeaturesProcessor(), SeasonalAnalysisProcessor()]
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
        
    @app.callback(
        Output('eda-statistics-content', 'children'),
        Output('eda-histogram-plot', 'figure'),
        Output('eda-boxplot-plot', 'figure'),
        Output('eda-time-series-plot', 'figure'),
        Input('eda-stat-dropdown', 'value'),
        Input('eda-hist-dropdown', 'value'),
        Input('eda-boxplot-dropdown', 'value'),
        Input('etl-data', 'data')
    )
    def update_eda(selected_stat, selected_hist, selected_box, stored_data):
        if stored_data is None:
            return "Carga los datos primero", {}, {}, {}
            
        json_data = io.StringIO(stored_data)
            
        df = pd.read_json(json_data, orient='split')
        
        # 1. Estadísticas Descriptivas
        stats = df[selected_stat].describe()
        stats_content = [
            html.P(f"Variable: {selected_stat}"),
            html.P(f"Media: {stats['mean']:.2f}"),
            html.P(f"Mediana: {stats['50%']:.2f}"),
            html.P(f"Desv. Estándar: {stats['std']:.2f}"),
            html.P(f"Mínimo: {stats['min']:.2f}"),
            html.P(f"Máximo: {stats['max']:.2f}"),
            html.P(f"Datos: {int(stats['count'])} registros")
        ]
        
        # 2. Generar Histograma
        hist_fig = px.histogram(
            df, 
            x=selected_hist,
            nbins=50,
            title=f'Distribución de {selected_hist}'
        )
        
        # 3. Generar Boxplot
        box_fig = px.box(
            df,
            y=selected_box,
            title=f'Distribución de {selected_box}'
        )
        
        # 4. Serie Temporal
        time_series = df.groupby('arrival_date').size().reset_index(name='reservas')
        time_fig = px.line(
            time_series,
            x='arrival_date',
            y='reservas',
            title='Evolución Temporal de Reservas'
        )
        
        return stats_content, hist_fig, box_fig, time_fig

    # Sincronización de Dropdowns
    @app.callback(
        Output('eda-hist-dropdown', 'value'),
        Output('eda-boxplot-dropdown', 'value'),
        Input('eda-stat-dropdown', 'value')
    )
    def sync_dropdowns(selected_stat):
        return selected_stat, selected_stat
    
    
    # Callback para conexión a DB
    @app.callback(
        Output('output-db-connection', 'children'),
        Output('store-data', 'data', allow_duplicate=True),
        Input('db-connect-btn', 'n_clicks'),
        State('db-host', 'value'),
        State('db-port', 'value'),
        State('db-name', 'value'),
        State('db-user', 'value'),
        State('db-password', 'value'),
        State('db-table', 'value'),
        prevent_initial_call=True
    )
    def load_from_db(n_clicks, host, port, dbname, user, password, tabla):
        if n_clicks:
            try:
                # Usar tu controlador para obtener datos
                dbManager = DatabaseManager()
                
                dbManager.carga_connect(
                    host=host,
                    database=dbname,
                    user=user,
                    password=password,
                    port=port
                )
                
                df = dbManager.load_from_db(tabla)
                
                preview = html.Div([
                    html.H5(f"Tabla cargada: {tabla}"),
                    html.P(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}"),
                    dash_table.DataTable(
                        data=df.head(10).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "minWidth": "100px"},
                        page_size=10,
                    )
                ])
                
                # Mostrar el preview del archivo
                preview = html.Div([
                    html.H5(f"Tabla cargada: {tabla}"),
                    html.P(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}"),
                    dash_table.DataTable(
                        data=df.head(10).to_dict('records'),
                        columns=[{"name": i, "id": i} for i in df.columns],
                        style_table={"overflowX": "auto"},
                        style_cell={"textAlign": "left", "minWidth": "100px"},
                        page_size=10,
                    )
                ])
                
                if df is not None:
                    return preview, df.to_json(date_format='iso', orient='split')
                    
            except Exception as e:
                return html.Div([
                    html.H5("Error de conexión", style={'color': 'red'}),
                    html.P(str(e))
                ])
        return None

    # Callback para descargas
    @app.callback(
        Output('download-data', 'data'),
        Input('download-csv-btn', 'n_clicks'),
        Input('download-json-btn', 'n_clicks'),
        Input('download-excel-btn', 'n_clicks'),
        State('etl-data', 'data'),
        prevent_initial_call=True
    )
    def handle_download(csv_clicks, json_clicks, excel_clicks, etl_data):
        ctx = dash.callback_context
        if not ctx.triggered or etl_data is None:
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        df = pd.read_json(etl_data, orient='split')

        try:
            if button_id == 'download-csv-btn':
                return dcc.send_data_frame(df.to_csv, "datos_procesados.csv", index=False)
            
            elif button_id == 'download-json-btn':
                return dcc.send_data_frame(df.to_json, "datos_procesados.json", orient='records', indent=2)
            
            elif button_id == 'download-excel-btn':
                return dcc.send_data_frame(df.to_excel, "datos_procesados.xlsx", index=False, engine='openpyxl')

        except Exception as e:
            print(f"Error en descarga: {str(e)}")
            return no_update

    # Callback para habilitar botones
    @app.callback(
        [Output('download-csv-btn', 'disabled'),
         Output('download-json-btn', 'disabled'),
        Output("toggle-export-form", "disabled"),
         Output('download-excel-btn', 'disabled')],
        Input('etl-data', 'data')
    )
    def toggle_buttons(etl_data):
        return [etl_data is None] * 4
    
    @app.callback(
        Output('output-data-upload', 'style'),
        Output('output-db-connection', 'style'),
        Input('last-data-source', 'data'),
    )
    def toggle_results_display(last_source):
        file_style = {'display': 'block'} if last_source == 'file' else {'display': 'none'}
        db_style = {'display': 'block'} if last_source == 'db' else {'display': 'none'}
        return file_style, db_style
    
    @app.callback(
        Output('last-data-source', 'data', allow_duplicate=True),
        Input('upload-data', 'contents'),
        prevent_initial_call=True
    )
    def set_last_source_file(_):
        return 'file'

    @app.callback(
        Output('last-data-source', 'data', allow_duplicate=True),
        Input('db-connect-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def set_last_source_db(_):
        return 'db'
    
    @app.callback(
    Output("export-form-collapse", "is_open"),
    Input("toggle-export-form", "n_clicks"),
    State("export-form-collapse", "is_open"),
    prevent_initial_call=True
)
    def toggle_export_form(n_clicks, is_open):
        if n_clicks:
            return not is_open
        return is_open
        
    @app.callback(
        Output("export-status", "children"),
        Input("export-to-db-btn", "n_clicks"),
        State("etl-data", "data"),
        State("db-host", "value"),
        State("db-port", "value"),
        State("db-name", "value"),
        State("db-user", "value"),
        State("db-password", "value"),
        State("db-table", "value"),
        prevent_initial_call=True
    )
    def export_to_postgres(n_clicks, etl_data, host, port, dbname, user, password, table_name):
        if not etl_data:
            return "No hay datos procesados para exportar."

        try:
            df = pd.read_json(etl_data, orient="split")
            db_manager = DatabaseManager()
            
            connected = db_manager.carga_connect(
                host=host,
                database=dbname,
                user=user,
                password=password,
                port=port
            )

            if not connected:
                return "Error al conectar a PostgreSQL. Revisa tus credenciales."

            saved = db_manager.save_to_db(df, table_name)

            if saved:
                return "Exportación exitosa a PostgreSQL."
            else:
                return "Error al guardar datos en la base de datos."

        except Exception as e:
            return f"Error al exportar: {str(e)}"