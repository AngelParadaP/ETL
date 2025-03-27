import numpy as np
import pandas as pd
import os
from abc import ABC, abstractmethod

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
        
# ----------------------------
# Sistema Principal
# ----------------------------
class HotelDataSystem:
    def __init__(self):
        self.processors = [
            DateProcessor(),    # Angel
            CleanProcessor(),   # Balam

        ]
    
    def load_data(self):
        """Carga datos desde archivo"""
        while True:
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

    def save_data(self, df):
        """Guarda los datos procesados"""
        while True:
            choice = input("Guardar como (1)CSV, (2)JSON, (3)Excel: ")
            path = input("Ruta de guardado: ")
            try:
                if choice == '1':
                    if not path.lower().endswith('.csv'):
                        path += '.csv'  # Fuerza extensión .csv
                    try:
                        df.to_csv(path, index=False)
                    except Exception as e:
                        print(f"Error al guardar CSV: {e}")
                        continue
                        
                elif choice == '2':
                    if not path.lower().endswith('.json'):
                        path += '.json'  # Fuerza extensión .json
                    try:
                        df.to_json(path, orient='records')
                    except Exception as e:
                        print(f"Error al guardar JSON: {e}")
                        continue
                        
                elif choice == '3':
                    if not path.lower().endswith('.xlsx'):
                        path += '.xlsx'  # Fuerza extensión .xlsx
                    try:
                        df.to_excel(path, index=False)
                    except Exception as e:
                        print(f"Error al guardar Excel: {e}")
                        continue
                else:
                    print("Opción inválida")
                    continue
                print("Datos guardados exitosamente!")
                break
            except Exception as e:
                print(f"Error al guardar: {e}")

    def run(self):
        """Ejecuta el sistema"""
        print("=== Sistema de Procesamiento de Datos Hotel ===")
        df = self.load_data()
        
        for processor in self.processors:
            df = processor.process(df)
        
        print("\nProcesamiento completado. Resumen:")
        print(df.info())
        self.save_data(df)

if __name__ == "__main__":
    system = HotelDataSystem()
    system.run()