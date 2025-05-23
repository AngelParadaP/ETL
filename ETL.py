import numpy as np
import pandas as pd
import psycopg2
from abc import ABC, abstractmethod
from psycopg2 import sql
from psycopg2.extras import execute_values
import getpass
from sklearn.preprocessing import MinMaxScaler

class DataProcessor(ABC):
    """
    Clase base para todos los procesadores de datos
    se crea para que todas las clases hereden de ella y tengan que implementar el metodo process
    """
    @abstractmethod
    def process(self, df):
        pass

class DateProcessor(DataProcessor):
    """
    Funciones Pandas únicas:
    1. pd.to_datetime() - Conversión de formatos de fecha
    2. dt.strftime() - Formateo de fechas
    3. pd.Timedelta() - Cálculo con diferencias de tiempo
    
    Objetivo: Estandarizar todos los formatos de fecha y calcular estadías
    """
    def process(self, df):
        """
        Parametros: 
            - df: DataFrame a formatear
            
        Retorna: 
            - df: DataFrame con fechas estandarizadas y estadías calculadas
        """
        try:
            # Función 1 
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
            
            # Función 2 
            df['arrival_date'] = pd.to_datetime(
                df['arrival_date_year'].astype(str) + '-' +
                df['arrival_date_month'] + '-' +
                df['arrival_date_day_of_month'].astype(str)
            ).dt.strftime('%Y-%m-%d')
            
            # Función 3 
            df['total_stay'] = (pd.to_datetime(df['reservation_status_date']) - 
                               pd.to_datetime(df['arrival_date'])).dt.days
            print("\nProcesamiento de fechas completado.\n")
            return df
        except Exception as e:
            print(f"Error en procesamiento de fechas: {e}")
            return df

class CleanProcessor(DataProcessor):
    """
    Funciones Pandas únicas:
    4. dropna() - Elimina filas con valores vacíos
    5. isna() - Detecta valores nulos
    6. duplicated() - Detecta instancias duplicadas basandose en todas las columnas o un subset
    7. drop_duplicates() - Elimina instancias duplicadas
    
    Objetivo: Eliminación rigurosa de datos incompletos o corruptos
    """
    def process(self, df):
        """
        Parametros: 
            - df: DataFrame a limpiar
            
        Retorna: 
            - df: DataFrame con filas con datos criticos vacíos o filas duplicadas eliminadas
        """
        
        try:
            df.replace([None, '', 'NULL', 'NaN', 'NaT', 'null'], np.nan, inplace=True)
            
            df['country'] = df['country'].str.strip().replace(r'^\s*$', np.nan, regex=True)
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
            df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
            
            initial_count = len(df)
            critical_columns = ['country', 'adr', 'reservation_status_date']
            df.dropna(subset=critical_columns, how='any', inplace=True)
                        # Eliminar filas con valores nulos en 'agent'
            if 'agent' in df.columns:
                df.dropna(subset=['agent'], inplace=True)

            # Eliminar columna 'company' si existe
            if 'company' in df.columns:
                df.drop(columns=['company'], inplace=True)
            # Consideraremos que un registro es duplicado si coincide en TODOS estos campos:
            duplicate_subset = [
                'hotel',
                'lead_time',  # Tiempo de anticipación único para cada reserva
                'arrival_date_year',
                'arrival_date_month',
                'arrival_date_day_of_month',
                'adults',
                'children',
                'babies',
                'reservation_status_date',  # Fecha única de la reserva
                'adr'  # Tarifa diaria promedio (poco probable que sea exactamente igual)
            ]

            # 1. Primero identificamos los duplicados reales
            duplicates = df.duplicated(subset=duplicate_subset, keep=False)
            
            
            df = df.drop_duplicates(subset=duplicate_subset, keep='first')
                        
            print(f"\nEliminacion de instancias con valores vacios o duplicadas completada:\n")
            print(f"Registros duplicados encontrados: {duplicates.sum()}")
            print(f"- Filas eliminadas: {initial_count - len(df)}")
            print(f"- Valores nulos restantes por columna:")
            print(df.isna().sum())
            
            return df
        except Exception as e:
            print(f"Error durante limpieza: {str(e)}")
            return df
        
class NormalizerProcessor(DataProcessor):
    """
    Aplica Min-Max normalization a columnas numéricas como 'adr', 'lead_time' y 'total_stay'
    """
    def __init__(self, columns=None):
        self.columns = columns or ['adr', 'lead_time', 'total_stay']

    def process(self, df):
        try:
            df = df.copy()
            scaler = MinMaxScaler()
            for col in self.columns:
                if col in df.columns:
                    df[f"{col}_norm"] = scaler.fit_transform(df[[col]])
            print("Normalización completada.")
            return df
        except Exception as e:
            print(f"Error en NormalizerProcessor: {e}")
            return df
        
class FilterProcessor(DataProcessor):
    """
    Filtra registros inválidos o irrelevantes, como:
    - Tarifas (adr) menores o iguales a 0
    - Reservas sin huéspedes
    """
    def process(self, df):
        try:
            df = df.copy()

            condiciones = (
                (df['adr'] > 0) &
                ((df['adults'] + df['children'].fillna(0) + df['babies'].fillna(0)) > 0)
            )

            registros_antes = len(df)
            df_filtrado = df[condiciones]
            registros_despues = len(df_filtrado)

            print(f"Filtrado aplicado. Filas eliminadas: {registros_antes - registros_despues}")
            return df_filtrado

        except Exception as e:
            print(f"Error en FilterProcessor: {e}")
            return df
        
class PredictiveFeaturesProcessor(DataProcessor):
    """
    Funciones Pandas únicas:
    8. pd.cut() - Discretización de variables continuas
    9. pd.get_dummies() - Codificación one-hot
    10. pd.qcut() - Creación de características agregadas
    
    Objetivo: Preparar características para modelo predictivo de cancelaciones
    """
    def process(self, df):
        """
        Parametros: 
            - df: DataFrame a procesar
            
        Retorna: 
            - df: DataFrame con características predictivas agregadas
        """
        try:

            # Validar columnas requeridas
            required_columns = {
                'lead_time', 'deposit_type', 'meal', 'customer_type', 
                'market_segment', 'adr', 'is_canceled',
                'stays_in_weekend_nights', 'stays_in_week_nights'
            }
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")
            
             # Funcion 1. Categorización de lead_time
            bins = [0, 7, 30, 90, 365, float('inf')]
            labels = ['Last minute (0-7)', 'Short term (8-30)', 
                    'Medium term (31-90)', 'Long term (91-365)', 
                    'Very early (>365)']
            
            df['lead_time_category'] = pd.cut(
                df['lead_time'],
                bins=bins,
                labels=labels,
                right=False
            ).astype(str)  # Convertir a string para mejor compatibilidad

            # Funcion 2. Codificación one-hot
            cols_to_encode = ['deposit_type', 'meal', 'customer_type', 'market_segment']
            
            encoded = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode, drop_first=False)
            encoded = encoded.astype(int)
            
            df = pd.concat([df, encoded], axis=1)

            # Funcion 3. Características agregadas 
            # Calcular tasa de cancelación por segmento (una sola vez)
            market_cancel_rate = df.groupby('market_segment')['is_canceled'].mean()
            df['market_segment_cancel_rate'] = df['market_segment'].map(market_cancel_rate)
            
            # Discretizar ADR 
            df['adr_quantile'] = pd.qcut(
                df['adr'], 
                q=4, 
                labels=['Q1', 'Q2', 'Q3', 'Q4'] 
            ).astype(str)
            
            # Total de noches
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            
            print("\nProcesamiento para el agregado de características predictivas completado.\n")
            
            return df
        except Exception as e:
            print(f"Error en preparación de características predictivas: {e}")
            return df
        
class SpecialRequestPredictorPrep(DataProcessor):
    """
    Crea una columna binaria 'many_requests' que indica si un cliente tiene más de 2 solicitudes especiales.
    """
    def process(self, df):
        try:
            if 'total_of_special_requests' in df.columns:
                df['many_requests'] = (df['total_of_special_requests'] > 2).astype(int)
                print("Columna 'many_requests' creada exitosamente.")
            else:
                print("La columna 'total_of_special_requests' no está disponible.")
            return df
        except Exception as e:
            print(f"Error al crear columna many_requests: {e}")
            return df        


class SeasonalAnalysisProcessor(DataProcessor):
    """
    Analiza estacionalidad y crea nuevas columnas:
    - season: Temporada del año (Invierno, Primavera, Verano, Otoño)
    - demand_period: Temporada de demanda (Alta, Media, Baja)
    
    Funciones pandas únicas:
    11. dt.month - Extrae el mes de la fecha
    12. pd.Series.map - Mapeo personalizado de valores
    13. dt.is_quarter_start - Identifica inicio de trimestre (para lógica de temporada)
    14. dt.to_period('Q') - Conversión a periodo trimestral (para análisis agregado)
    """
    def process(self, df):
        """
        Parametros: 
            - df: DataFrame a procesar
            
        Retorna: 
            - df: DataFrame con columnas de temporada y periodo de demanda
        """
        try:
            # Convertir a datetime si no lo está
            if not pd.api.types.is_datetime64_any_dtype(df['arrival_date']):
                df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
            
            # 1. Crear columna de temporada (season) usando mapeo condicional
            season_rules = {
                1: 'Winter', 2: 'Winter', 3: 'Spring',
                4: 'Spring', 5: 'Spring', 6: 'Summer',
                7: 'Summer', 8: 'Summer', 9: 'Autumn',
                10: 'Autumn', 11: 'Autumn', 12: 'Winter'
            }
            df['season'] = df['arrival_date'].dt.month.map(season_rules)
            
            # 2. Crear columna de periodo de demanda (demand_period) 
            # Usando inicio de trimestre (is_quarter_start) como parte de la lógica
            def get_demand_period(date):
                # Usamos is_quarter_start para identificar cambios estacionales importantes
                is_q_start = date.is_quarter_start
                
                # Usamos to_period para agrupar por trimestres
                quarter = date.to_period('Q').quarter
                
                # Lógica mejorada que considera el trimestre y si es inicio
                if quarter in [2, 3] or (quarter == 4 and date.month == 12 and date.day > 15):
                    return 'High'
                elif (quarter in [1, 4] and not is_q_start) or (quarter == 2 and date.month == 5):
                    return 'Normal'
                else:
                    return 'Low'
            
            df['demand_period'] = df['arrival_date'].apply(get_demand_period)
            
            # 3. Análisis agregado por trimestre usando to_period
            df['arrival_quarter'] = df['arrival_date'].dt.to_period('Q')
            quarter_stats = df.groupby('arrival_quarter').agg({
                'is_canceled': ['count', 'mean'],
                'adr': 'mean'
            }).reset_index()
            
            quarter_stats.columns = ['quarter', 'total_reservations', 'cancellation_rate', 'avg_daily_rate']
            
            print("\nProcesamiento estacional y de temporada completado")
            
            return df.drop('arrival_quarter', axis=1)  # Eliminamos la columna temporal
        
        except Exception as e:
            print(f"Error en análisis estacional: {e}")
            return df



class DatabaseManager:
    """Manejador de conexión y operaciones con PostgreSQL"""
    def __init__(self):
        self.connection = None
        
    def carga_connect(self, host, database, user, password, port):
        try:
            self.connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
            print("¡Conexión exitosa a PostgreSQL!")
            return True
        except Exception as e:
            print(f"Error al conectar a PostgreSQL: {e}")
            return False
    
    def connect(self):
        """Establece conexión con PostgreSQL"""
        print("\nConfiguración de conexión a PostgreSQL:")
        host = input("Host (localhost): ") or "localhost"
        database = input("Nombre de la base de datos: ")
        user = input("Usuario: ")
        password = getpass.getpass("Contraseña: ")
        port = input("Puerto (5432): ") or "5432"
        
        try:
            self.connection = psycopg2.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port
            )
            print("¡Conexión exitosa a PostgreSQL!")
            return True
        except Exception as e:
            print(f"Error al conectar a PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Cierra la conexión con la base de datos"""
        if self.connection:
            self.connection.close()
            print("Conexión a PostgreSQL cerrada.")
    
    def save_to_db(self, df, table_name):
        """Guarda el DataFrame en una tabla de PostgreSQL"""
        if not self.connection:
            print("No hay conexión a la base de datos.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Crear tabla si no existe
            columns = []
            for col, dtype in df.dtypes.items():
                if pd.api.types.is_integer_dtype(dtype):
                    sql_type = "INTEGER"
                elif pd.api.types.is_float_dtype(dtype):
                    sql_type = "FLOAT"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    sql_type = "TIMESTAMP"
                elif pd.api.types.is_bool_dtype(dtype):
                    sql_type = "BOOLEAN"
                else:
                    sql_type = "TEXT"
                
                columns.append(sql.Identifier(col) + sql.SQL(" ") + sql.SQL(sql_type))
            
            create_table_query = sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    {}
                )
            """).format(
                sql.Identifier(table_name),
                sql.SQL(",\n    ").join(columns)
            )
            
            cursor.execute(create_table_query)
            self.connection.commit()
            
            # Insertar datos
            columns = [sql.Identifier(col) for col in df.columns]
            values = [sql.Placeholder()] * len(df.columns)
            
            insert_query = sql.SQL("""
                INSERT INTO {} ({})
                VALUES %s
                ON CONFLICT DO NOTHING
            """).format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(columns)
            )
            
            # Convertir DataFrame a lista de tuplas
            data_tuples = [tuple(x) for x in df.to_numpy()]
            
            execute_values(cursor, insert_query, data_tuples)
            self.connection.commit()
            
            print(f"Datos guardados exitosamente en la tabla '{table_name}'")
            return True
        except Exception as e:
            print(f"Error al guardar en PostgreSQL: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def load_from_db(self, table_name):
        """Carga datos desde una tabla de PostgreSQL"""
        if not self.connection:
            print("No hay conexión a la base de datos.")
            return None
        
        try:
            cursor = self.connection.cursor()
            
            query = sql.SQL("SELECT * FROM {}").format(
                sql.Identifier(table_name)
            )
            
            cursor.execute(query)
            columns = [desc[0] for desc in cursor.description]
            data = cursor.fetchall()
            
            df = pd.DataFrame(data, columns=columns)
            print(f"Datos cargados exitosamente desde la tabla '{table_name}\n")
            return df
        except Exception as e:
            print(f"Error al cargar desde PostgreSQL: {e}")
            return None
        finally:
            if cursor:
                cursor.close()

# ----------------------------
# Sistema Principal
# ----------------------------
class HotelDataSystem:
    def __init__(self):
        self.processors = [
            DateProcessor(),
            CleanProcessor(),
            NormalizerProcessor(),
            FilterProcessor(),
            SeasonalAnalysisProcessor(),
            SpecialRequestPredictorPrep(),
            PredictiveFeaturesProcessor()
        ]
        self.db_manager = DatabaseManager()
    
    def load_data(self):
        """
        Parametros: 
            - Ninguno
            
        Retorna: 
            - df: DataFrame basado en el archivo de datos cargado
            
        Carga datos desde archivo o base de datos
        
        Funciones unicas de Pandas:
        
        16. pd.read_csv() - Carga CSV
        17. pd.read_json() - Carga JSON
        18. pd.read_excel() - Carga Excel
        
        """
        while True:
            print("\nOpciones de carga:")
            print("1. Desde archivo (CSV/JSON/XLSX)")
            print("2. Desde PostgreSQL")
            print("3. Salir")
            choice = input("Seleccione opción: ")
            
            if choice == '1':
                file_path = input("Ingrese ruta del archivo (CSV/JSON/XLSX): ")
                try:
                    if file_path.endswith('.csv'):
                        return pd.read_csv(file_path)
                    elif file_path.endswith('.json'):
                        return pd.read_json(file_path)
                    elif file_path.endswith(('.xlsx', '.xls')):
                        return pd.read_excel(file_path)
                    else:
                        print("Formato no soportado. Use CSV, JSON o XLSX.")
                except Exception as e:
                    print(f"Error al cargar: {e}. Intente nuevamente.")
            elif choice == '2':
                if self.db_manager.connect():
                    table_name = input("Nombre de la tabla a cargar: ")
                    df = self.db_manager.load_from_db(table_name)
                    if df is not None:
                        return df
            elif choice == '3':
                return None
            else:
                print("Opción inválida. Intente nuevamente.")

    def save_data(self, df):
        """
        Parametros: 
            - df: DataFrame a guardar en un archivo
            
        Retorna: 
            - Nada
            
        Guarda los datos procesados
        
        Funciones únicas de Pandas:
        19. DataFrame.to_csv() - Guarda como CSV
        20. DataFrame.to_json() - Guarda como JSON
        21. DataFrame.to_excel() - Guarda como Excel
        
        """
        while True:
            print("\nOpciones de guardado:")
            print("1. CSV")
            print("2. JSON")
            print("3. Excel")
            print("4. PostgreSQL")
            print("5. Salir")
            choice = input("Seleccione una opcion: ")
            
            if choice in ('1', '2', '3'):
                path = input("Ruta de guardado: ")
                try:
                    if choice == '1':
                        if not path.lower().endswith('.csv'):
                            path += '.csv'
                        df.to_csv(path, index=False)
                    elif choice == '2':
                        if not path.lower().endswith('.json'):
                            path += '.json'
                        df.to_json(path, orient='records')
                    elif choice == '3':
                        if not path.lower().endswith('.xlsx'):
                            path += '.xlsx'
                        df.to_excel(path, index=False)
                    input("Datos guardados exitosamente!")
                except Exception as e:
                    print(f"Error al guardar: {e}")
            elif choice == '4':
                if not self.db_manager.connection and not self.db_manager.connect():
                    continue
                table_name = input("Nombre de la tabla destino: ")
                if self.db_manager.save_to_db(df, table_name):
                    pass
            elif choice == '5':
                print("Saliendo del sistema...")
                return
            else:
                input("Opción inválida")

    def run(self):
        """Ejecuta el sistema"""
        print("=== Sistema de Procesamiento de Datos Hotel ===")
        df = self.load_data()
        
        if df is None or df.empty:
            print("No se pudieron cargar datos. Saliendo...")
            return
        
        for processor in self.processors:
            df = processor.process(df)
        
        print("\nProcesamiento completado. Resumen:")
        print(df.info())
        
        self.save_data(df)
        
        # Cerrar conexión a la base de datos si está abierta
        self.db_manager.disconnect()

if __name__ == "__main__":
    system = HotelDataSystem()
    system.run()
