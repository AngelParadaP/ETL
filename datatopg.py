import pandas as pd
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
import getpass

class RawDatabaseManager:
    """Manejador de conexión y operaciones con PostgreSQL para datos crudos"""
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
    
    def save_raw_data(self, df, table_name):
        """Guarda el DataFrame en PostgreSQL SIN procesamiento"""
        if not self.connection:
            print("No hay conexión a la base de datos.")
            return False
        
        try:
            cursor = self.connection.cursor()
            
            # Crear tabla con tipos de datos genéricos para preservar formato original
            columns = []
            for col in df.columns:
                # Usamos TEXT para todo para preservar el formato original
                columns.append(sql.Identifier(col) + sql.SQL(" TEXT"))
            
            create_table_query = sql.SQL("""
                DROP TABLE IF EXISTS {};
                CREATE TABLE {} (
                    {}
                )
            """).format(
                sql.Identifier(table_name),
                sql.Identifier(table_name),
                sql.SQL(",\n    ").join(columns)
            )
            
            cursor.execute(create_table_query)
            self.connection.commit()
            
            # Insertar datos tal cual están
            columns = [sql.Identifier(col) for col in df.columns]
            insert_query = sql.SQL("""
                INSERT INTO {} ({})
                VALUES %s
            """).format(
                sql.Identifier(table_name),
                sql.SQL(", ").join(columns)
            )
            
            # Convertir DataFrame a lista de tuplas manteniendo valores exactos
            data_tuples = [tuple(str(x) if not pd.isna(x) else None for x in row) 
                         for row in df.itertuples(index=False)]
            
            execute_values(cursor, insert_query, data_tuples)
            self.connection.commit()
            
            print(f"\nDatos crudos guardados exitosamente en la tabla '{table_name}'")
            print(f"- Total de registros: {len(df)}")
            print(f"- Columnas: {df.columns.tolist()}")
            
            return True
        except Exception as e:
            print(f"Error al guardar datos crudos: {e}")
            self.connection.rollback()
            return False
        finally:
            if cursor:
                cursor.close()

class RawHotelDataLoader:
    """Cargador de datos hotelero sin procesamiento"""
    def __init__(self):
        self.db_manager = RawDatabaseManager()
    
    def load_raw_file(self):
        """Carga archivo sin modificar datos"""
        while True:
            file_path = input("\nIngrese ruta del archivo (CSV/JSON/XLSX): ")
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, keep_default_na=True)
                elif file_path.endswith('.json'):
                    df = pd.read_json(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_path)
                else:
                    print("Formato no soportado. Use CSV, JSON o XLSX.")
                    continue
                
                print("\nResumen de datos crudos:")
                print(df.head())
                print("\nValores nulos por columna:")
                print(df.isna().sum())
                
                return df
            except Exception as e:
                print(f"Error al cargar archivo: {e}")
    
    def run(self):
        """Ejecuta el cargador de datos crudos"""
        print("=== Cargador de Datos Hotelero (Sin Procesamiento) ===")
        
        if not self.db_manager.connect():
            return
        
        df = self.load_raw_file()
        if df is None or df.empty:
            print("No se pudieron cargar datos. Saliendo...")
            return
        
        table_name = input("\nNombre de la tabla destino en PostgreSQL: ")
        self.db_manager.save_raw_data(df, table_name)
        
        self.db_manager.disconnect()

if __name__ == "__main__":
    loader = RawHotelDataLoader()
    loader.run()