import numpy as np
import pandas as pd
import psycopg2
from abc import ABC, abstractmethod
from psycopg2 import sql
from psycopg2.extras import execute_values
import getpass

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
            # Solo procesar si la columna no existe
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
            
            if 'arrival_date' not in df.columns:  # Solo crear si no existe
                df['arrival_date'] = pd.to_datetime(
                    df['arrival_date_year'].astype(str) + '-' +
                    df['arrival_date_month'] + '-' +
                    df['arrival_date_day_of_month'].astype(str)
                ).dt.strftime('%Y-%m-%d')
            
            if 'total_stay' not in df.columns:  # Solo crear si no existe
                df['total_stay'] = (pd.to_datetime(df['reservation_status_date']) - 
                                  pd.to_datetime(df['arrival_date'])).dt.days
            print("\nProcesamiento de fechas completado.\n", flush=True)
            return df
        except Exception as e:
            print(f"Error en procesamiento de fechas: {e}", flush=True)
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
                        
            print(f"\nEliminacion de instancias con valores vacios o duplicadas completada:\n", flush=True)
            print(f"Registros duplicados encontrados: {duplicates.sum()}", flush=True)
            print(f"- Filas eliminadas: {initial_count - len(df)}", flush=True)
            print(f"- Valores nulos restantes por columna:", flush=True)
            print(df.isna().sum(), flush=True)
            
            return df
        except Exception as e:
            print(f"Error durante limpieza: {str(e)}", flush=True)
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
            required_columns = {
                'lead_time', 'deposit_type', 'meal', 'customer_type', 
                'market_segment', 'adr', 'is_canceled',
                'stays_in_weekend_nights', 'stays_in_week_nights'
            }
            missing_columns = required_columns - set(df.columns)
            if missing_columns:
                raise ValueError(f"Columnas requeridas faltantes: {missing_columns}")
            
            # Solo crear si no existe
            if 'lead_time_category' not in df.columns:
                bins = [0, 7, 30, 90, 365, float('inf')]
                labels = ['Last minute (0-7)', 'Short term (8-30)', 
                        'Medium term (31-90)', 'Long term (91-365)', 
                        'Very early (>365)']
                
                df['lead_time_category'] = pd.cut(
                    df['lead_time'],
                    bins=bins,
                    labels=labels,
                    right=False
                ).astype(str)

            # Solo hacer codificación one-hot si las columnas no existen
            cols_to_encode = ['deposit_type', 'meal', 'customer_type', 'market_segment']
            new_cols = [f"{col}_{val}" for col in cols_to_encode for val in df[col].unique()]
            
            if not any(col in df.columns for col in new_cols):
                encoded = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode, drop_first=False)
                encoded = encoded.astype(int)
                df = pd.concat([df, encoded], axis=1)

            # Solo crear si no existe
            if 'market_segment_cancel_rate' not in df.columns:
                market_cancel_rate = df.groupby('market_segment')['is_canceled'].mean()
                df['market_segment_cancel_rate'] = df['market_segment'].map(market_cancel_rate)
            
            if 'adr_quantile' not in df.columns:
                df['adr_quantile'] = pd.qcut(
                    df['adr'], 
                    q=4, 
                    labels=['Q1', 'Q2', 'Q3', 'Q4'] 
                ).astype(str)
            
            if 'total_nights' not in df.columns:
                df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            
            print("\nProcesamiento para el agregado de características predictivas completado.\n", flush=True)
            
            return df
        except Exception as e:
            print(f"Error en preparación de características predictivas: {e}", flush=True)
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
            if not pd.api.types.is_datetime64_any_dtype(df['arrival_date']):
                df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')
            
            # Solo crear si no existe
            if 'season' not in df.columns:
                season_rules = {
                    1: 'Winter', 2: 'Winter', 3: 'Spring',
                    4: 'Spring', 5: 'Spring', 6: 'Summer',
                    7: 'Summer', 8: 'Summer', 9: 'Autumn',
                    10: 'Autumn', 11: 'Autumn', 12: 'Winter'
                }
                df['season'] = df['arrival_date'].dt.month.map(season_rules)
            
            # Solo crear si no existe
            if 'demand_period' not in df.columns:
                def get_demand_period(date):
                    is_q_start = date.is_quarter_start
                    quarter = date.to_period('Q').quarter
                    
                    if quarter in [2, 3] or (quarter == 4 and date.month == 12 and date.day > 15):
                        return 'High'
                    elif (quarter in [1, 4] and not is_q_start) or (quarter == 2 and date.month == 5):
                        return 'Normal'
                    else:
                        return 'Low'
                
                df['demand_period'] = df['arrival_date'].apply(get_demand_period)
            
            # Esta columna es temporal y se elimina al final, no necesita verificación
            df['arrival_quarter'] = df['arrival_date'].dt.to_period('Q')
            quarter_stats = df.groupby('arrival_quarter').agg({
                'is_canceled': ['count', 'mean'],
                'adr': 'mean'
            }).reset_index()
            
            quarter_stats.columns = ['quarter', 'total_reservations', 'cancellation_rate', 'avg_daily_rate']
            
            print("\nProcesamiento estacional y de temporada completado", flush=True)
            print("\nEstadísticas por trimestre:", flush=True)
            print(quarter_stats.to_string(index=False), flush=True)
            
            return df.drop('arrival_quarter', axis=1)
        
        except Exception as e:
            print(f"Error en análisis estacional: {e}", flush=True)
            return df



class DatabaseManager:
    """Manejador de conexión y operaciones con PostgreSQL"""
    def __init__(self):
        self.connection = None
    
    def connect(self):
        """Establece conexión con PostgreSQL"""
        print("\nConfiguración de conexión a PostgreSQL:", flush=True)
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
            print("¡Conexión exitosa a PostgreSQL!", flush=True)
            return True
        except Exception as e:
            print(f"Error al conectar a PostgreSQL: {e}", flush=True)
            return False
    
    def disconnect(self):
        """Cierra la conexión con la base de datos"""
        if self.connection:
            self.connection.close()
            print("Conexión a PostgreSQL cerrada.", flush=True)
    
    def save_to_db(self, df, table_name):
        """Guarda el DataFrame en una tabla de PostgreSQL"""
        if not self.connection:
            print("No hay conexión a la base de datos.", flush=True)
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
            
            print(f"Datos guardados exitosamente en la tabla '{table_name}'", flush=True)
            return True
        except Exception as e:
            print(f"Error al guardar en PostgreSQL: {e}", flush=True)
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()
    
    def load_from_db(self, table_name):
        """Carga datos desde una tabla de PostgreSQL"""
        if not self.connection:
            print("No hay conexión a la base de datos.", flush=True)
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
            print(f"Datos cargados exitosamente desde la tabla '{table_name}\n", flush=True)
            return df
        except Exception as e:
            print(f"Error al cargar desde PostgreSQL: {e}", flush=True)
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
            SeasonalAnalysisProcessor(),
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
            print("\nOpciones de carga:", flush=True)
            print("1. Desde archivo (CSV/JSON/XLSX)", flush=True)
            print("2. Desde PostgreSQL", flush=True)
            print("3. Salir", flush=True)
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
                        print("Formato no soportado. Use CSV, JSON o XLSX.", flush=True)
                except Exception as e:
                    print(f"Error al cargar: {e}. Intente nuevamente.", flush=True)
            elif choice == '2':
                if self.db_manager.connect():
                    table_name = input("Nombre de la tabla a cargar: ")
                    df = self.db_manager.load_from_db(table_name)
                    if df is not None:
                        return df
            elif choice == '3':
                return None
            else:
                print("Opción inválida. Intente nuevamente.", flush=True)

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
            print("\nOpciones de guardado:", flush=True)
            print("1. CSV", flush=True)
            print("2. JSON", flush=True)
            print("3. Excel", flush=True)
            print("4. PostgreSQL", flush=True)
            print("5. Salir", flush=True)
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
                    print(f"Error al guardar: {e}", flush=True)
            elif choice == '4':
                if not self.db_manager.connection and not self.db_manager.connect():
                    continue
                table_name = input("Nombre de la tabla destino: ")
                if self.db_manager.save_to_db(df, table_name):
                    pass
            elif choice == '5':
                print("Saliendo del sistema...", flush=True)
                return
            else:
                input("Opción inválida")

    def run(self):
        """Ejecuta el sistema"""
        print("=== Sistema de Procesamiento de Datos Hotel ===", flush=True)
        df = self.load_data()
        
        if df is None or df.empty:
            print("No se pudieron cargar datos. Saliendo...", flush=True)
            return
        
        for processor in self.processors:
            df = processor.process(df)
        
        print("\nProcesamiento completado. Resumen:", flush=True)
        print(df.info(), flush=True)
        
        self.save_data(df)
        
        # Cerrar conexión a la base de datos si está abierta
        self.db_manager.disconnect()

if __name__ == "__main__":
    system = HotelDataSystem()
    system.run()
