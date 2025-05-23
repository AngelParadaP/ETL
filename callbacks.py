from dash import Input, Output, State, html, dash_table, dcc, no_update
import pandas as pd
from ETL import DateProcessor, CleanProcessor, NormalizerProcessor, FilterProcessor, PredictiveFeaturesProcessor, SpecialRequestPredictorPrep, SeasonalAnalysisProcessor
import base64
import io
import plotly.express as px
from ETL import DatabaseManager
import dash
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import plotly.figure_factory as ff
import numpy as np
import json
import dash_bootstrap_components as dbc
from dash import ALL
from utils import generar_tabla

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
            df = pd.read_excel(io.BytesIO(decoded), dtype=str)
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
        generar_tabla(df, alto='600px')
    ])
    return df, preview

# Función principal para registrar todos los callbacks
def register_callbacks(app):
    # ==================== CALLBACKS ====================
    
    # Callback para cargar el archivo
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

        df_original = pd.read_json(json_data, orient='split')
        df_etl = df_original.copy()

        filas_iniciales = len(df_etl)
        columnas_iniciales = set(df_etl.columns)

        secciones = []

        # === Paso 1: Fechas ===
        try:
            df_antes = df_etl.copy()

            # Guardar columna original antes del cambio 
            fecha_col_antes = df_antes['reservation_status_date'].copy() if 'reservation_status_date' in df_antes.columns else None

            df_etl = DateProcessor().process(df_etl)

            # Contar cuántas fechas válidas hay ahora (después de to_datetime)
            fechas_convertidas = df_etl['reservation_status_date'].notna().sum() if 'reservation_status_date' in df_etl.columns else 0

            resumen = f"""
    Conversión de Fechas:
    - Se convirtieron correctamente {fechas_convertidas} fechas a formato datetime.
    - Se creó la columna 'total_stay'.
    """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en DateProcessor: {e}", color="danger"))

        # === Paso 2: Limpieza ===
        try:
            df_antes = df_etl.copy()

            filas_antes = len(df_antes)
            nulos_antes = df_antes.isna().sum().sum()

            df_etl = CleanProcessor().process(df_etl)

            filas_despues = len(df_etl)
            nulos_despues = df_etl.isna().sum().sum()

            filas_eliminadas = filas_antes - filas_despues
            resumen = f"""
    Limpieza de Datos:
    - Se eliminaron {filas_eliminadas} filas (nulos o duplicados).
    - Nulos antes: {int(nulos_antes)} | Nulos después: {int(nulos_despues)}.
    """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en CleanProcessor: {e}", color="danger"))

        # === Paso 3: Normalización ===
        try:
            df_antes = df_etl.copy()
            columnas_antes = set(df_antes.columns)

            df_etl = NormalizerProcessor().process(df_etl)

            columnas_despues = set(df_etl.columns)
            nuevas_columnas = columnas_despues - columnas_antes

            resumen = f"""
        Normalización:
        - Se aplicó Min-Max normalization a las columnas numéricas seleccionadas.
        - Columnas añadidas: {', '.join(nuevas_columnas) if nuevas_columnas else 'ninguna'}.
        """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en NormalizerProcessor: {e}", color="danger"))

        # === Paso 4: Filtrado de registros inválidos ===
        try:
            df_antes = df_etl.copy()
            filas_antes = len(df_antes)

            df_etl = FilterProcessor().process(df_etl)

            filas_despues = len(df_etl)
            filas_eliminadas = filas_antes - filas_despues

            resumen = f"""
        Filtrado de Registros:
        - Se eliminaron {filas_eliminadas} filas con valores inválidos.
        - Criterios: adr > 0 y al menos 1 huésped (adulto, niño o bebé).
        """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en FilterProcessor: {e}", color="danger"))

        # === Paso 5: Variables predictivas ===
        try:
            df_antes = df_etl.copy()
            columnas_antes = set(df_antes.columns)

            df_etl = PredictiveFeaturesProcessor().process(df_etl)
            columnas_despues = set(df_etl.columns)

            nuevas_columnas = columnas_despues - columnas_antes

            resumen = f"""
    Agregado de Características:
    - Se agregaron {len(nuevas_columnas)} nuevas columnas al DataFrame.
    """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en PredictiveFeaturesProcessor: {e}", color="danger"))

        # === Paso 6: Clasificación de clientes exigentes ===
        try:
            df_antes = df_etl.copy()

            df_etl = SpecialRequestPredictorPrep().process(df_etl)

            resumen = f"""
    Clasificación de Clientes Exigentes:
    - Se creó la columna 'many_requests' (1 si el cliente tiene más de 2 solicitudes especiales).
    - Valores 1 (exigente): {df_etl['many_requests'].sum()} registros.
    """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en SpecialRequestPredictorPrep: {e}", color="danger"))

        # === Paso 7: Análisis Estacional ===
        try:
            df_antes = df_etl.copy()
            columnas_antes = set(df_antes.columns)

            df_etl = SeasonalAnalysisProcessor().process(df_etl)
            columnas_despues = set(df_etl.columns)

            nuevas_columnas = columnas_despues - columnas_antes

            resumen = f"""
    Análisis Estacional:
    - Columnas añadidas: {', '.join(nuevas_columnas) if nuevas_columnas else 'ninguna'}.
    """
            secciones.append(dbc.Alert(html.Pre(resumen.strip()), color="light"))

            secciones.append(html.P("Antes:"))
            secciones.append(dash_table.DataTable(
                data=df_antes.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_antes.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.P("Después:"))
            secciones.append(dash_table.DataTable(
                data=df_etl.head(3).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_etl.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'minWidth': '80px'},
                page_size=3
            ))
            secciones.append(html.Hr())
        except Exception as e:
            secciones.append(dbc.Alert(f"Error en SeasonalAnalysisProcessor: {e}", color="danger"))

        # === Resumen final ===
        filas_finales = len(df_etl)
        columnas_finales = len(df_etl.columns)
        resumen_final = f"""
    ✅ Resumen Final del ETL:
    - Filas al inicio: {filas_iniciales}
    - Filas después del ETL: {filas_finales}
    - Columnas al inicio: {len(columnas_iniciales)}
    - Columnas al final: {columnas_finales}
    """
        secciones.append(dbc.Alert(html.Pre(resumen_final.strip()), color="success"))

        return html.Div(secciones), df_etl.to_json(date_format='iso', orient='split')
    
    #Callback para filtros interactivos
    @app.callback(
        Output('filtro-pais', 'options'),
        Input('etl-data', 'data')
    )
    def actualizar_dropdown_paises(json_data):
        if not json_data:
            return []
        df = pd.read_json(json_data, orient='split')
        paises = df['country'].dropna().unique()
        return [{'label': p, 'value': p} for p in sorted(paises)]
    
    # Callback para aplicar filtros
    @app.callback(
        Output('filtro-etl-output', 'children'),
        Input('filtro-aplicar', 'n_clicks'),
        State('etl-data', 'data'),
        State('filtro-pais', 'value'),
        State('filtro-adr', 'value'),
        prevent_initial_call=True
    )
    def aplicar_filtro_etl(n_clicks, json_data, pais, adr_min):
        if not json_data:
            return html.Div("No hay datos transformados para filtrar.")

        df = pd.read_json(json_data, orient='split')

        if pais:
            df = df[df['country'] == pais]
        if adr_min is not None:
            df = df[df['adr'] >= adr_min]

        if df.empty:
            return html.Div("⚠️ No hay registros que coincidan con los filtros.")

        return html.Div([
            html.H6(f"{len(df)} filas coinciden con los filtros aplicados:"),
            dash_table.DataTable(
                data=df.head(10).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                page_size=10,
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'}
            )
        ])
        
    # Callback para el análisis exploratorio de datos (EDA)
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

    # Sincronización de Dropdowns para EDA
    @app.callback(
        Output('eda-hist-dropdown', 'value'),
        Output('eda-boxplot-dropdown', 'value'),
        Input('eda-stat-dropdown', 'value')
    )
    def sync_dropdowns(selected_stat):
        return selected_stat, selected_stat
    
    # Callback para conexión a base de datos
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
                
                if df is not None:
                    return preview, df.to_json(date_format='iso', orient='split')
                    
            except Exception as e:
                return html.Div([
                    html.H5("Error de conexión", style={'color': 'red'}),
                    html.P(str(e))
                ])
        return None

    # Callback para descargas de datos
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

    # Callback para habilitar botones de descarga
    @app.callback(
        [Output('download-csv-btn', 'disabled'),
         Output('download-json-btn', 'disabled'),
         Output("toggle-export-form", "disabled"),
         Output('download-excel-btn', 'disabled')],
        Input('etl-data', 'data')
    )
    def toggle_buttons(etl_data):
        return [etl_data is None] * 4
    
    # Callbacks para manejar la fuente de datos (archivo vs base de datos)
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
    
    # Callback para el formulario de exportación
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
        
    # Callback para exportar a PostgreSQL
    @app.callback(
        Output("export-status", "children"),
        Input("export-to-db-btn", "n_clicks"),
        State("etl-data", "data"),
        State("db-hosteo", "value"),
        State("db-puerto", "value"),
        State("db-nombre", "value"),
        State("db-usuario", "value"),
        State("db-contra", "value"),
        State("db-tabla", "value"),
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
        










    # ==================== CALLBACKS DE MINERÍA DE DATOS ====================
    
    # Callback para el clustering de clientes
    @app.callback(
        [Output('cluster-plot', 'figure'),
        Output('cluster-boxplot', 'figure'),
        Output('cluster-insights', 'children'),
        Output('cluster-table', 'data'),
        Output('cluster-table', 'columns')],
        [Input('run-clustering', 'n_clicks')],
        [State('cluster-features', 'value'),
        State('n-clusters', 'value'),
        State('etl-data', 'data')],
        prevent_initial_call=True
    )
    def run_clustering(n_clicks, features, n_clusters, etl_data):
        if etl_data is None:
            return (
                dash.no_update, 
                dash.no_update, 
                html.Div("Por favor carga los datos primero", style={'color': 'red'}),
                dash.no_update,
                dash.no_update
            )
        
        if n_clicks is None or not features or len(features) < 2:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        try:
            df = pd.read_json(etl_data, orient='split')
            if df.empty:
                return (
                    dash.no_update, 
                    dash.no_update, 
                    html.Div("Los datos están vacíos", style={'color': 'red'}),
                    dash.no_update,
                    dash.no_update
                )
            
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            df['total_revenue'] = df['adr'] * df['total_nights']
            
            X = df[features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['cluster'] = kmeans.fit_predict(X_scaled)
            df['cluster'] = df['cluster'].astype(str)
            
            if len(features) == 2:
                fig = px.scatter(
                    df.sample(min(5000, len(df))),
                    x=features[0], 
                    y=features[1], 
                    color='cluster',
                    hover_data=['hotel', 'customer_type', 'total_revenue'],
                    title=f"Segmentación de Clientes ({n_clusters} clusters)"
                )
            else:
                fig = px.scatter_3d(
                    df.sample(min(5000, len(df))),
                    x=features[0], 
                    y=features[1], 
                    z=features[2],
                    color='cluster',
                    hover_data=['hotel', 'customer_type', 'total_revenue'],
                    title=f"Segmentación de Clientes ({n_clusters} clusters)"
                )
            
            box_fig = px.box(
                df, 
                x='cluster', 
                y='total_revenue',
                color='cluster',
                title='Distribución de Ingresos por Cluster'
            )
            
            # Flatten MultiIndex columns for DataTable
            cluster_stats = df.groupby('cluster').agg({
                'lead_time': ['mean', 'median'],
                'total_nights': ['mean', 'median'],
                'adr': ['mean', 'median'],
                'total_revenue': ['mean', 'median', 'count'],
                'is_canceled': 'mean'
            }).reset_index()
            # Flatten columns
            cluster_stats.columns = ['_'.join([str(i) for i in col if i]).strip('_') for col in cluster_stats.columns.values]
            
            insights_list = []
            for cluster in range(n_clusters):
                cluster_str = str(cluster)
                cluster_data = df[df['cluster'] == cluster_str]
                avg_lead = cluster_data['lead_time'].mean()
                avg_nights = cluster_data['total_nights'].mean()
                avg_adr = cluster_data['adr'].mean()
                cancel_rate = cluster_data['is_canceled'].mean()
                
                insights_list.append(html.Div([
                    html.H5(f"Cluster {cluster_str}"),
                    html.P(f"Clientes con lead time promedio de {avg_lead:.1f} días, estadía de {avg_nights:.1f} noches"),
                    html.P(f"Tarifa diaria promedio: ${avg_adr:.2f} - Tasa de cancelación: {cancel_rate:.1%}"),
                    html.P(f"Total clientes: {len(cluster_data)} ({len(cluster_data)/len(df):.1%})"),
                    html.Hr()
                ]))
            
            insights = html.Div(insights_list)
            
            table_data = cluster_stats.to_dict('records')
            columns = [{"name": col, "id": col} for col in cluster_stats.columns]
            
            # Return figures directly, not .to_dict()
            return fig, box_fig, insights, table_data, columns
        
        except Exception as e:
            print(f"Error en clustering: {str(e)}")
            return (
                dash.no_update, 
                dash.no_update, 
                html.Div(f"Error al procesar los datos: {str(e)}", style={'color': 'red'}),
                dash.no_update,
                dash.no_update
            )
        
    #función para variables dinámicas de predicción        

    def generar_interfaz_prediccion(features, df):
        prediction_inputs = []

        for feature in features:
            input_id = {'type': 'pred-input', 'index': feature}

            if df[feature].dtype == 'object' or feature in ['hotel', 'deposit_type']:
                if feature in ['hotel', 'deposit_type']:
                    opciones = {
                        'hotel': ['Resort Hotel', 'City Hotel'],
                        'deposit_type': ['No Deposit', 'Refundable', 'Non Refund']
                    }[feature]
                else:
                    opciones = df[feature].dropna().unique()

                prediction_inputs.append(
                    dbc.Row([
                        dbc.Col(dbc.Label(feature), width=4),
                        dbc.Col(
                            dcc.Dropdown(
                                id=input_id,
                                options=[{'label': opt, 'value': opt} for opt in opciones],
                                placeholder=f"Selecciona {feature}"
                            ),
                            width=8
                        )
                    ], className="mb-2")
                )
            else:
                prediction_inputs.append(
                    dbc.Row([
                        dbc.Col(dbc.Label(feature), width=4),
                        dbc.Col(
                            dbc.Input(
                                id=input_id,
                                type='number',
                                placeholder=f"Ingresar {feature}"
                            ),
                            width=8
                        )
                    ], className="mb-2")
                )

        return html.Div([
            html.H5("Simular Nuevo Cliente"),
            html.Div(prediction_inputs),
            dbc.Button("Predecir Cliente Exigente", id="predict-btn", color="primary", className="mt-3"),
            html.Div(id='prediction-result')
        ])


    # Callback para el modelo de predicción de clientes exigentes
    @app.callback(
        [Output('model-performance', 'children'),
         Output('feature-importance', 'figure'),
         Output('confusion-matrix', 'figure'),
         Output('model-rules', 'children'),
         Output('prediction-interface', 'children'),
         Output('mining-models', 'data')],
        [Input('train-model', 'n_clicks')],
        [State('prediction-features', 'value'),
         State('model-type', 'value'),
         State('etl-data', 'data')],
         prevent_initial_call=True
    )
    def train_exigente_model(n_clicks, features, model_type, etl_data):
        if n_clicks is None or not features:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        df = pd.read_json(etl_data, orient='split')
        
        # Preprocesamiento
        X = pd.get_dummies(df[features])
        y = df['many_requests']

        # Solo aplicar SMOTE si es Árbol de Decisión y Regresión lógistica
        if model_type in ['decision_tree', 'logistic']:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)
        else:
            X_res, y_res = X, y
        
        # Entrenar modelo
        if model_type == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # logistic
            model = LogisticRegression(max_iter=1000, random_state=42)
        
        model.fit(X_res, y_res)
        
        # Evaluar modelo
        y_pred = model.predict(X)
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().reset_index()
        
        # Importancia de características
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:  # Para regresión logística
            importance = np.abs(model.coef_[0])
        
        feature_imp = pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        imp_fig = px.bar(
            feature_imp.head(10),
            x='importance',
            y='feature',
            title='Importancia de Variables para Predecir Clientes Exigentes',
        )
        
        # Matriz de confusión
        cm = confusion_matrix(y, y_pred)
        cm_fig = ff.create_annotated_heatmap(
            z=cm,
            x=['No Exigente', 'Exigente'],
            y=['No Exigente', 'Exigente'],
            colorscale='Blues',
            showscale=True
        )
        cm_fig.update_layout(title='Matriz de Confusión')
        
        # Reglas del modelo (solo para árboles)
        rules = ""
        if model_type == 'decision_tree':
            rules = export_text(model, feature_names=list(X.columns))
            rules = html.Pre(rules)
        else:
            rules = html.P("Este modelo no genera reglas explícitas. Ver importancia de características arriba.")
        
        # Interfaz de predicción
        prediction_interface = generar_interfaz_prediccion(features, df)
        
        # Guardar modelo serializado
        model_data = {
            'model_type': model_type,
            'features': features,
            'model_params': json.dumps(model.get_params()),
            'feature_names': list(X.columns)
        }
        
        # Mostrar métricas de desempeño
        metrics = html.Div([
            html.H4("Desempeño del Modelo"),
            dash_table.DataTable(
                data=report_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in report_df.columns],
                style_table={'overflowX': 'auto'}
            ),
            html.P(f"Modelo utilizado: {model_type.replace('_', ' ').title()}")
        ])
        
        return metrics, imp_fig, cm_fig, rules, prediction_interface, model_data
    
    
    # Callback para predicción en tiempo real
    @app.callback(
        Output('prediction-result', 'children'),
        Input('predict-btn', 'n_clicks'),
        State('mining-models', 'data'),
        State('etl-data', 'data'),
        State('prediction-features', 'value'),
        State({'type': 'pred-input', 'index': ALL}, 'value'),
        prevent_initial_call=True
    )
    def predict_exigente(n_clicks, model_data, etl_data, features, input_values):
        if n_clicks is None:
            return dash.no_update

        if not model_data or not input_values:
            return dbc.Alert("Faltan datos del modelo o entradas de usuario.", color="danger")

        if len(input_values) != len(features):
            return dbc.Alert(
                f"Se esperaban {len(features)} valores, pero se recibieron {len(input_values)}.",
                color="danger"
            )

        try:
            # Convertir los datos ingresados en un DataFrame
            user_input = pd.DataFrame([input_values], columns=features)

            # Restaurar datos de entrenamiento para asegurar mismo preprocesamiento
            df = pd.read_json(etl_data, orient='split')
            X_train = pd.get_dummies(df[features])
            
            # Aplicar mismo one-hot encoding
            user_input_dummies = pd.get_dummies(user_input)
            user_input_dummies = user_input_dummies.reindex(columns=X_train.columns, fill_value=0)

            # Restaurar el modelo (tipo y parámetros)
            model_type = model_data['model_type']
            model_params = json.loads(model_data['model_params'])

            if model_type == 'decision_tree':
                model = DecisionTreeClassifier(**model_params)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(**model_params)
            else:
                model = LogisticRegression(**model_params)

            # Reentrenar modelo
            y_train = df['many_requests']
            model.fit(X_train, y_train)

            # Realizar predicción
            prob = model.predict_proba(user_input_dummies)[0][1]

            return html.Div([
                html.H5("Resultado de la Predicción"),
                dbc.Alert(
                    f"Probabilidad de que el cliente sea exigente (many_requests=1): {prob:.1%}",
                     color="warning" if prob > 0.5 else "success"
                ),
                html.P("Valores ingresados:"),
                html.Pre(str(user_input.to_dict('records')))
            ])
        
        except Exception as e:
            return dbc.Alert(f"Error al predecir: {str(e)}", color="danger")

    
    # Callback para el análisis de ingresos
    @app.callback(
        [Output('revenue-regression', 'figure'),
         Output('revenue-heatmap', 'figure'),
         Output('revenue-insights', 'children'),
         Output('pricing-recommendations', 'children')],
        [Input('analyze-revenue', 'n_clicks')],
        [State('revenue-target', 'value'),
         State('revenue-features', 'value'),
         State('etl-data', 'data')]
    )
    def analyze_revenue(n_clicks, target, features, etl_data):
        if n_clicks is None or not features:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update
            
        df = pd.read_json(etl_data, orient='split')
        
        # Crear variables derivadas
        df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
        df['total_revenue'] = df['adr'] * df['total_nights']
        
        # Gráfico de regresión
        reg_fig = px.scatter(
            df, x=features[0], y=target,
            trendline="ols",
            color='arrival_date_month',
            title=f"Relación entre {features[0]} y {target}"
        )
        
        # Heatmap de correlaciones
        numeric_cols = df.select_dtypes(include=['number']).columns
        corr = df[numeric_cols].corr()
        heat_fig = px.imshow(
            corr,
            labels=dict(color="Correlación"),
            title="Correlación entre Variables Numéricas"
        )
        
        # Insights basados en los datos
        high_season = df.groupby('arrival_date_month')['adr'].mean().idxmax()
        low_season = df.groupby('arrival_date_month')['adr'].mean().idxmin()
        room_type_price = df.groupby('reserved_room_type')['adr'].mean().sort_values(ascending=False)
        
        insights = html.Div([
            html.H5("Hallazgos Clave:"),
            html.Ul([
                html.Li(f"Temporada alta: {high_season} (ADR promedio: ${df[df['arrival_date_month']==high_season]['adr'].mean():.2f})"),
                html.Li(f"Temporada baja: {low_season} (ADR promedio: ${df[df['arrival_date_month']==low_season]['adr'].mean():.2f})"),
                html.Li(f"Habitación más cara: {room_type_price.index[0]} (${room_type_price[0]:.2f} por noche)"),
                html.Li(f"Clientes recurrentes pagan {df[df['is_repeated_guest']==1]['adr'].mean() - df[df['is_repeated_guest']==0]['adr'].mean():.2f} más en promedio")
            ])
        ])
        
        # Recomendaciones de precios
        recommendations = html.Div([
            html.H5("Recomendaciones:"),
            html.Ul([
                html.Li(f"Aumentar precios en {high_season} durante reservas de última hora (lead time < 7 días)"),
                html.Li(f"Ofrecer paquetes para {low_season} que incluyan noches extra con descuento"),
                html.Li("Crear programa de fidelidad para clientes recurrentes con tarifas preferenciales"),
                html.Li("Implementar precios dinámicos basados en demanda histórica por tipo de habitación")
            ])
        ])
        
        return reg_fig, heat_fig, insights, recommendations
    
    
    #================= CALLBACK DESICIONES =================
    @app.callback(
        Output('decision-content', 'children'),
        Input('etl-data', 'data')
    )
    def render_decision_content(etl_json):
        if not etl_json:
            return dbc.Alert("No hay datos disponibles.", color="warning")

        df = pd.read_json(io.StringIO(etl_json), orient='split')

        if 'cluster' in df.columns:
            fig = px.scatter(
                df,
                x='lead_time',
                y='adr',
                color='cluster',
                title='Clusters de Clientes (Lead Time vs ADR)',
                labels={'lead_time': 'Días de Anticipación', 'adr': 'Tarifa Diaria Promedio'}
            )
        else:
            fig = px.scatter(
                df,
                x='lead_time',
                y='adr',
                color='market_segment' if 'market_segment' in df.columns else None,
                title='Relación entre Anticipación y Precio',
                labels={'lead_time': 'Lead Time', 'adr': 'ADR'}
            )

        return dbc.Container([
            html.H4("Toma de Decisiones", className="my-4"),

            dbc.Row([
                dbc.Col([
                    html.H5("🎯 Objetivo del análisis"),
                    html.P("Identificar perfiles de clientes más rentables y predecir comportamientos clave, como las solicitudes especiales, para tomar decisiones estratégicas sobre precios, promociones y operación del hotel.")
                ])
            ]),

            html.Hr(),

            dbc.Row([
                dbc.Col([
                    html.H5("📌 Hallazgos Relevantes"),
                    html.Ul([
                        html.Li("🧩 El Cluster 2 representa solo el 22.9% de los clientes, pero genera los mayores ingresos con un ADR promedio de $181.10."),
                        html.Li("🕒 El Cluster 1 reserva con 193 días de anticipación, pero tiene una tasa alta de cancelación (39.9%)."),
                        html.Li("📉 El Cluster 0 es el más grande (51.8%), pero con tarifas bajas y corta estancia."),
                    ])
                ], width=6),

                dbc.Col([
                    html.H5("📊 Visualización de Apoyo"),
                    dcc.Graph(figure=fig)
                ], width=6)
            ]),

            dbc.Row([
                dbc.Col([
                    html.H5("🧠 Segmentación de Clientes (Clustering)", className="mt-4"),
                    html.Ul([
                        html.Li("📈 Cluster 2: ADR promedio de $181.10 — más rentable (22.9% de clientes)"),
                        html.Li("⏳ Cluster 1: Anticipación promedio de 193 días — alto riesgo de cancelación (39.9%)"),
                        html.Li("📉 Cluster 0: Clientes frecuentes pero con tarifas bajas — base operativa del negocio"),
                    ])
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.H5("💸 Análisis de Precios e Ingresos", className="mt-4"),
                    html.Ul([
                        html.Li("🌞 Temporada alta: Agosto — ADR promedio de $154.87"),
                        html.Li("❄️ Temporada baja: Enero — ADR promedio de $76.24"),
                        html.Li("🏨 Habitación más cara: Tipo H — $193.78 por noche"),
                        html.Li("🔁 Clientes recurrentes pagan $31.39 menos en promedio")
                    ])
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.H5("🔍 Hallazgos del Análisis Exploratorio de Datos (EDA)", className="mt-4"),
                    html.Ul([
                        html.Li("📉 La mayoría de los clientes reservan con menos de 60 días de anticipación."),
                        html.Li("💰 La tarifa diaria (ADR) presenta outliers importantes por encima de $5000."),
                        html.Li("🛏️ La mayoría de las estancias duran entre 1 y 5 noches."),
                        html.Li("📅 Se observan picos de reservas en verano, especialmente en julio y agosto."),
                    ])
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.H5("✅ Recomendaciones Estratégicas", className="mt-4"),
                    html.Ul([
                        html.Li("🎯 Dirigir promociones premium al Cluster 2 para maximizar ingresos."),
                        html.Li("🔒 Aplicar políticas de cancelación estrictas a clientes con alta anticipación (Cluster 1)."),
                        html.Li("📦 Hacer upselling al Cluster 0 con paquetes que aumenten su valor."),
                        html.Li("📊 Aumentar precios en agosto para reservas de último minuto (alta demanda)."),
                        html.Li("🎁 Ofrecer noches extra con descuento en enero para mejorar ocupación."),
                        html.Li("🏅 Crear programa de fidelidad con beneficios exclusivos para clientes recurrentes."),
                        html.Li("🛎️ Anticipar servicios adicionales para clientes exigentes y ajustar operaciones con base en la predicción."),
                        html.Li("🔍 Validar y controlar outliers en ADR para evitar distorsiones en decisiones de pricing."),
                        html.Li("📆 Programar recursos operativos en función de la estacionalidad observada."),
                    ])
                ])
            ])
        ])
