import numpy as np
import pandas as pd
import os

try:
    file_path = "hotel_bookings.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: El archivo {file_path} no existe en la ubicación actual.")
        print(f"Directorio actual: {os.getcwd()}")
        print("Por favor coloca el archivo en esta ubicación o especifica la ruta completa.")
    else:
        df = pd.read_csv(file_path)
        # Agregar datos vacíos
        df.loc[10:15, "country"] = np.nan  # Vaciar la columna "country" en algunas filas

        # Agregar formatos inconsistentes en fechas
        df.loc[20:25, "reservation_status_date"] = "2023/15/10"  # Formato incorrecto

        # Guardar en distintos formatos
        df.to_csv("hotel_bookings_modified.csv", index=False)
        df.to_excel("hotel_bookings_modified.xlsx", index=False)
        df.to_json("hotel_bookings_modified.json", orient="records")  
              
except FileNotFoundError:
    print("Error: Archivo no encontrado. Verifica la ruta y nombre del archivo.")
except Exception as e:
    print(f"Ocurrió un error: {str(e)}")