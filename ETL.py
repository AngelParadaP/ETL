import numpy as np
import pandas as pd
import os
import psycopg2
from abc import ABC, abstractmethod
from psycopg2 import sql
from psycopg2.extras import execute_values
import getpass
from matplotlib import pyplot as plt

class DataProcessor(ABC):
    @abstractmethod
    def process(self, df):
        pass


# --- Angel: Procesamiento de Fechas ---
class DateProcessor(DataProcessor):
    """
    Funciones Pandas únicas:
    1. pd.to_datetime() - Conversión de formatos de fecha
    2. dt.strftime() - Formateo de fechas
    3. pd.Timedelta() - Cálculo con diferencias de tiempo
    
    Objetivo: Estandarizar todos los formatos de fecha y calcular estadías
    """
    def process(self, df):
        try:
            # Función 1 (única para Angel)
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
            
            # Función 2 (única)
            df['arrival_date'] = pd.to_datetime(
                df['arrival_date_year'].astype(str) + '-' +
                df['arrival_date_month'] + '-' +
                df['arrival_date_day_of_month'].astype(str)
            ).dt.strftime('%Y-%m-%d')
            
            # Función 3 (única)
            df['total_stay'] = (pd.to_datetime(df['reservation_status_date']) - 
                               pd.to_datetime(df['arrival_date'])).dt.days
            return df
        except Exception as e:
            print(f"Error en procesamiento de fechas: {e}")
            return df

class ReservationGrouping(DataProcessor):
    """
    Funciones Pandas únicas:
    1. pd.to_datetime() - Conversión a tipo fecha
    2. dt.to_period() - Extracción de periodo (mes-año)
    3. pivot_table() - Agrupamiento multidimensional
    
    Objetivo: Analizar reservas mensuales por tipo de hotel y visualizar tendencias
    """
    def process(self, df):
        try:
            # Función 1 (única): Convertir a datetime
            df['arrival_date'] = pd.to_datetime(df['arrival_date'])
            
            # Función 2 (única): Crear periodo año-mes
            arrival_year_month = df['arrival_date'].dt.to_period('M')
            
            # Función 3 (única): Crear tabla pivote para agrupamiento
            reservas_por_mes = df.pivot_table(
                index=arrival_year_month,
                columns='hotel',
                values='is_canceled',  # Usamos cualquier columna para contar
                aggfunc='count',
                fill_value=0
            )
            
            # Generar visualización
            reservas_por_mes.plot(
                kind='bar',
                figsize=(12, 6),
                title='Reservas por Mes y Tipo de Hotel',
                xlabel='Mes de Llegada',
                ylabel='Número de Reservas'
            )
            
            plt.show()
            
            return df
        except Exception as e:
            print(f"Error en agrupamiento de reservas: {e}")
            return df
    

class CleanProcessor(DataProcessor):
    """
    Funciones Pandas únicas:
    1. dropna() - Elimina filas con valores vacíos
    2. isna() - Detecta valores nulos
    
    Objetivo: Eliminación rigurosa de datos incompletos o corruptos
    """
    def process(self, df):
        try:
            df.replace([None, '', 'NULL', 'NaN', 'NaT', 'null'], np.nan, inplace=True)
            
            df['country'] = df['country'].str.strip().replace(r'^\s*$', np.nan, regex=True)
            df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'], errors='coerce')
            df['adr'] = pd.to_numeric(df['adr'], errors='coerce')
            
            initial_count = len(df)
            critical_columns = ['country', 'adr', 'reservation_status_date']
            df.dropna(subset=critical_columns, how='any', inplace=True)
            
            print(f"\nLimpieza completada:")
            print(f"- Filas eliminadas: {initial_count - len(df)}")
            print(f"- Valores nulos restantes por columna:")
            print(df.isna().sum())
            
            return df
        except Exception as e:
            print(f"Error durante limpieza: {str(e)}")
            return df
        
class PredictiveFeaturesProcessor(DataProcessor):
    """
    Funciones Pandas únicas:
    1. pd.cut() - Discretización de variables continuas
    2. pd.get_dummies() - Codificación one-hot
    3. pd.qcut() + groupby().transform() - Creación de características agregadas
    
    Objetivo: Preparar características para modelo predictivo de cancelaciones
    """
    def process(self, df):
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
            

            # Eliminar filas con valores nulos en columnas críticas
            critical_cols = ['lead_time', 'adr']
            initial_rows = len(df)
            df.dropna(subset=critical_cols, inplace=True)
            if len(df) < initial_rows:
                print(f"Advertencia: Se eliminaron {initial_rows - len(df)} filas con valores nulos en columnas críticas")
            
            # Función 1 
            # Propósito: convertir la variable lead_time (tiempo de anticipación de la reserva) en categorías que puedan ser más útiles para el modelo predictivo.
            
            # Definir bins para categorizar el lead time
            bins = [0, 7, 30, 90, 365, float('inf')]
            labels = ['Último momento (0-7)', 'Corto plazo (8-30)', 
                    'Mediano plazo (31-90)', 'Largo plazo (91-365)', 
                    'Muy anticipado (>365)']
            
            df['lead_time_category'] = pd.cut(
                df['lead_time'],
                bins=bins,
                labels=labels,
                right=False
            )

            # funcion 2
            # Propósito: Convertir variables categóricas como deposit_type y meal en formato numérico para el modelo.

            # Pre-calcular antes de eliminar market_segment
            market_cancel_rate = df.groupby('market_segment')['is_canceled'].mean()
            
            # Columnas importantes para predecir cancelaciones
            cols_to_encode = ['deposit_type', 'meal', 'customer_type', 'market_segment']
            encoded = pd.get_dummies(df[cols_to_encode], prefix=cols_to_encode, drop_first=True)
            
            df = pd.concat([df, encoded], axis=1) 

            # Mapear tasas de cancelación antes de eliminar
            df['market_segment_cancel_rate'] = df['market_segment'].map(market_cancel_rate)
            df.drop(columns=cols_to_encode, inplace=True)
           
            # funcion 3
            # Propósito: Crear características agregadas como el "ADR promedio por tipo de cliente" que pueden ser predictores importantes.

             # Discretizar ADR (tarifa diaria promedio) en cuantiles
            df['adr_quantile'] = pd.qcut(df['adr'], q=4, labels=False)
            
            # Calcular tasa de cancelación promedio por segmento de mercado
            market_cancel_rate = df.groupby('market_segment')['is_canceled'].mean()
            df['market_segment_cancel_rate'] = df['market_segment'].map(market_cancel_rate)
            
            # Calcular días totales de estadía
            df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
            
            return df
        except Exception as e:
            print(f"Error en preparación de características predictivas: {e}")
            return df


class SeasonalAnalysisProcessor(DataProcessor):
    """
    Procesador de análisis estacional.

    Funciones Pandas únicas:
    1. .dt.month_name() - Extrae el nombre del mes desde fechas.
    2. groupby().size() - Cuenta cuántas reservas hubo por mes.
    3. sort_values() - Ordena los resultados por cantidad de reservas.

    Objetivo: Identificar temporadas altas y bajas de reservaciones.
    """
    def process(self, df):
        try:
            print("\n- Análisis de reservas por mes")
            
            # Asegurarse de que arrival_date sea tipo datetime
            if not pd.api.types.is_datetime64_any_dtype(df['arrival_date']):
                df['arrival_date'] = pd.to_datetime(df['arrival_date'], errors='coerce')

            # Función 1: Extraer nombre del mes
            df['arrival_month'] = df['arrival_date'].dt.month_name()

            # Función 2: Agrupar por mes y contar reservas
            reservas_por_mes = df.groupby('arrival_month').size().reset_index(name='total_reservas')

            # Función 3: Ordenar por número de reservas
            reservas_por_mes = reservas_por_mes.sort_values('total_reservas', ascending=False)
            reservas_por_mes = reservas_por_mes.reset_index(drop=True)

            print(reservas_por_mes)

            """"
            # Gráfico
            reservas_por_mes.plot(
                kind='bar',
                x='arrival_month',
                y='total_reservas',
                title='Reservas por Mes (Temporadas)',
                xlabel='Mes',
                ylabel='Total de Reservas',
                figsize=(10, 5)
            )
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("reservas_por_mes.png")
            plt.close()
            """

            return df
        except Exception as e:
            print(f"Error en análisis estacional: {e}")
            return df


class DatabaseManager:
    """Manejador de conexión y operaciones con PostgreSQL"""
    def __init__(self):
        self.connection = None
    
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
            print(f"Datos cargados exitosamente desde la tabla '{table_name}'")
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
            DateProcessor(),    # Angel
            CleanProcessor(),   # Balam
            SeasonalAnalysisProcessor(),    # Diana
        ]
        self.predictors = [
            PredictiveFeaturesProcessor() #Héctor
        ]
        
        self.analizers = [
            ReservationGrouping(),  # Balam
        ]
        self.db_manager = DatabaseManager()
    
    def load_data(self):
        """Carga datos desde archivo o base de datos"""
        while True:
            print("\nOpciones de carga:")
            print("1. Desde archivo (CSV/JSON/XLSX)")
            print("2. Desde PostgreSQL")
            choice = input("Seleccione opción (1/2): ")
            
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
            else:
                print("Opción inválida. Intente nuevamente.")

    def save_data(self, df):
        """Guarda los datos procesados"""
        while True:
            print("\nOpciones de guardado:")
            print("1. CSV")
            print("2. JSON")
            print("3. Excel")
            print("4. PostgreSQL")
            choice = input("Seleccione formato (1/2/3/4): ")
            
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
                    print("Datos guardados exitosamente!")
                    break
                except Exception as e:
                    print(f"Error al guardar: {e}")
            elif choice == '4':
                if not self.db_manager.connection and not self.db_manager.connect():
                    continue
                table_name = input("Nombre de la tabla destino: ")
                if self.db_manager.save_to_db(df, table_name):
                    break
            else:
                print("Opción inválida")

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
        
        save_data_option = 5
        exit_option = 6
        
        while True:
            print("\nAnalisis:")
            print("1. Cantidad de reservaciones por mes\n")
            print(f"{save_data_option}. Guardar datos")
            print(f"{exit_option}. Salir")
            
            try:
                analyzer_option = int(input("\nEscoja una opción: "))
            except ValueError:
                input("Ingrese un input valido")
                continue
            
            if analyzer_option == exit_option:
                break
            elif analyzer_option == save_data_option:
                self.save_data(df)
            elif analyzer_option not in range(1, len(self.analizers) + 1):
                input("Ingrese una opcion valida")
                continue
            else:
                self.analizers[analyzer_option - 1].process(df)

        
        # Cerrar conexión a la base de datos si está abierta
        self.db_manager.disconnect()

if __name__ == "__main__":
    system = HotelDataSystem()
    system.run()
