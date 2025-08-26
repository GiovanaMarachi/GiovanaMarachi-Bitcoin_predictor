import pandas as pd
import numpy as np

# --- Función para limpiar un solo DataFrame ---
def clean_crypto_data(df, file_name):
    print(f"\n--- Limpiando datos de {file_name} ---")

    # 1. Chequear tipos de datos iniciales y nulos
    print(f"\nInformación inicial de {file_name}:")
    df.info()

    # 2. Convertir columnas de tiempo (timestamp en milisegundos a datetime)
    print(f"\nConvirtiendo columnas de tiempo en {file_name}...")
    time_columns = ['timeOpen', 'timeClose', 'timeHigh', 'timeLow']
    for col in time_columns:
        if col in df.columns:
            # Dividir por 1000 para convertir milisegundos a segundos si es necesario
            # Usar errors='coerce' para convertir valores no válidos en NaT (Not a Time)
            df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
            if df[col].isnull().any():
                print(f"¡Atención! Algunas fechas en '{col}' no pudieron ser convertidas y se marcaron como NaT.")
                print(df[df[col].isnull()])
        else:
            print(f"Columna '{col}' no encontrada en {file_name}. Saltando conversión.")

    # 3. Asegurar que las columnas numéricas sean numéricas y manejar inconsistencias básicas
    price_volume_columns = ['priceOpen', 'priceHigh', 'priceLow', 'priceClose', 'volume']
    for col in price_volume_columns:
        if col in df.columns:
            # Convertir a numérico, forzando errores a NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                print(f"¡Atención! Valores no numéricos encontrados en '{col}' de {file_name} y convertidos a NaN.")
                print(df[df[col].isnull()])
            
            # Refactorización: Chequear si hay valores negativos en precios o volumen.
            invalid_values = df[df[col] < 0]
            if not invalid_values.empty:
                value_type = "precios" if 'price' in col else "volúmenes"
                print(f"¡Ojo! Se encontraron {value_type} negativos en '{col}' de {file_name}. Se convertirán a NaN.")
                df.loc[df[col] < 0, col] = np.nan
        else:
            print(f"Columna '{col}' no encontrada en {file_name}. Saltando verificación.")

    # 4. Verificar y manejar datos faltantes después de las conversiones
    print(f"\nChequeando datos faltantes en {file_name} después de conversiones y correcciones...")
    missing_after_conversion = df.isnull().sum()
    print(missing_after_conversion[missing_after_conversion > 0])

    # Opciones para manejar los datos faltantes:
    # Para datos de series de tiempo, a menudo se prefiere rellenar con el valor anterior
    # o eliminar las filas si la cantidad de datos faltantes es mínima y no afecta la continuidad.
    if missing_after_conversion.sum() > 0:
        print(f"\n¡Ojo! Todavía hay datos faltantes en {file_name}. Consideraciones:")
        print(f"- Si la cantidad es pequeña y no crucial para la continuidad, se pueden eliminar filas: `df.dropna(inplace=True)`")
        print(f"- Si querés rellenar, para series de tiempo, podrías usar `df.fillna(method='ffill', inplace=True)` (forward fill) o `df.fillna(df.mean(), inplace=True)` (con la media, cuidado con esto para precios).")
        # Por simplicidad y seguridad, eliminamos las filas con cualquier NaN en columnas críticas.
        # Definir columnas críticas para la eliminación de NaN.
        critical_columns = ['timeOpen', 'priceOpen', 'priceClose']
        initial_rows = len(df)
        df.dropna(subset=critical_columns, inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"Se eliminaron {rows_dropped} filas con NaNs en columnas críticas.")
        else:
            print("No se eliminaron filas con NaNs en columnas críticas.")
    else:
        print(f"¡Mba'eichapa! No hay datos faltantes en {file_name} después de la limpieza inicial.")

    # 5. Opcional: Eliminar duplicados si aplica (puede no ser común en datos de tiempo)
    # df.drop_duplicates(inplace=True)

    print(f"\n--- Primeras filas del DataFrame limpio de {file_name} ---")
    print(df.head())
    print(f"\nInformación final de {file_name}:")
    df.info()
    return df

def main():
    """
    Función principal para cargar, limpiar y guardar datos de criptomonedas.
    """
    files_to_process = [
        {'input': 'data/ethereum.xlsx', 'output': 'ethereum_limpio.xlsx', 'name': 'Ethereum'},
        {'input': 'data/bitcoin.xlsx', 'output': 'bitcoin_limpio.xlsx', 'name': 'Bitcoin'}
    ]

    for file_info in files_to_process:
        input_path = file_info['input']
        output_path = file_info['output']
        crypto_name = file_info['name']
        
        print(f"\n{'='*20} Procesando {crypto_name} {'='*20}")
        try:
            # ¡Cambio clave! Usar read_excel para archivos .xlsx
            # Asegúrate de tener 'openpyxl' instalado: pip install openpyxl
            # Forzamos el uso del motor 'openpyxl' para mayor seguridad.
            df = pd.read_excel(input_path, engine='openpyxl')
            df_cleaned = clean_crypto_data(df.copy(), crypto_name)
            df_cleaned.to_excel(output_path, index=False)
            print(f"\nArchivo '{output_path}' guardado con éxito.")
        except FileNotFoundError:
            print(f"¡Opá! No se encontró el archivo '{input_path}'. Asegúrate de que esté en la misma carpeta que el script.")
        except Exception as e:
            print(f"¡Pucha! Ocurrió un error inesperado al procesar '{input_path}': {e}")

if __name__ == "__main__":
    main()