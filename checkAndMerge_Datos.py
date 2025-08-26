import pandas as pd
import numpy as np
import os
import zipfile
import io

# --- Función para limpiar un solo DataFrame ---
def clean_crypto_data(df, file_name):
    print(f"\n--- Limpiando datos de {file_name} ---")

    # 1. Chequear tipos de datos iniciales y nulos
    print(f"\nInformación inicial de {file_name}:")
    df.info()

    # Identificar la columna de timestamp, que ahora se llama 'timestamp'
    timestamp_col = 'timestamp'
    if timestamp_col in df.columns:
        print(f"\nConvirtiendo columna de tiempo '{timestamp_col}' en {file_name}...")
        # Convertir de segundos (unidad='s') a datetime, forzando errores a NaT
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s', errors='coerce')
        if df[timestamp_col].isnull().any():
            print(f"¡Atención! Algunas fechas en '{timestamp_col}' no pudieron ser convertidas.")
    else:
        print(f"Columna '{timestamp_col}' no encontrada en {file_name}. Saltando conversión.")

    # 3. Asegurar que las columnas numéricas sean numéricas y manejar inconsistencias
    price_volume_columns = ['open', 'high', 'low', 'close', 'basevolume', 'usdtvolume']
    for col in price_volume_columns:
        if col in df.columns:
            # Convertir a numérico, forzando errores a NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Chequear si hay valores negativos en precios o volumen.
            if df[col].lt(0).any(): # `lt(0)` es una forma de chequear si es menor que 0
                print(f"¡Ojo! Se encontraron valores negativos en '{col}' de {file_name}. Se convertirán a NaN.")
                df.loc[df[col].lt(0), col] = np.nan
        else:
            print(f"Columna '{col}' no encontrada en {file_name}. Saltando verificación.")

    # 4. Verificar y manejar datos faltantes después de las conversiones
    print(f"\nChequeando datos faltantes en {file_name} después de conversiones y correcciones...")
    missing_after_conversion = df.isnull().sum()
    missing_after_conversion = missing_after_conversion[missing_after_conversion > 0]
    if not missing_after_conversion.empty:
        print(missing_after_conversion)
        print(f"\n¡Ojo! Todavía hay datos faltantes en {file_name}. Eliminando filas con NaNs en columnas críticas.")
        # Definir columnas críticas para la eliminación de NaN.
        critical_columns = ['timestamp', 'open', 'close']
        initial_rows = len(df)
        df.dropna(subset=critical_columns, inplace=True)
        rows_dropped = initial_rows - len(df)
        if rows_dropped > 0:
            print(f"Se eliminaron {rows_dropped} filas con NaNs en columnas críticas.")
        else:
            print("No se eliminaron filas con NaNs en columnas críticas.")
    else:
        print(f"¡Mba'eichapa! No hay datos faltantes en {file_name} después de la limpieza inicial.")

    print(f"\n--- Primeras filas del DataFrame limpio de {file_name} ---")
    print(df.head())
    print(f"\nInformación final de {file_name}:")
    df.info()
    return df

def main():
    """
    Función principal para cargar, limpiar y unir múltiples archivos XLSX
    que están dentro de archivos ZIP.
    """
    # Define la ruta de la carpeta que contiene los archivos ZIP
    # Aquí es donde tenés que poner el nombre de la carpeta, por ejemplo 'archivos_cripto'
    # Así si tus archivos están en una carpeta llamada 'data' y dentro otra 'Bitcoin'
    zip_folder = os.path.join('data', 'Bitcoin')

    # Lista de archivos ZIP a procesar (rango de un año)
    zip_filenames = [
        '20240618.zip', '20240619.zip', '20240620.zip', '20240621.zip', '20240622.zip',
        '20240623.zip', '20240624.zip', '20240625.zip', '20240626.zip', '20240627.zip',
        '20240628.zip', '20240629.zip', '20240630.zip', '20240701.zip', '20240702.zip',
        '20240703.zip', '20240704.zip', '20240705.zip', '20240706.zip', '20240707.zip',
        '20240708.zip', '20240709.zip', '20240710.zip', '20240711.zip', '20240712.zip',
        '20240713.zip', '20240714.zip', '20240715.zip', '20240716.zip', '20240717.zip',
        '20240718.zip', '20240719.zip', '20240720.zip', '20240721.zip', '20240722.zip',
        '20240723.zip', '20240724.zip', '20240725.zip', '20240726.zip', '20240727.zip',
        '20240728.zip', '20240729.zip', '20240730.zip', '20240731.zip', '20240801.zip',
        '20240802.zip', '20240803.zip', '20240804.zip', '20240805.zip', '20240806.zip',
        '20240807.zip', '20240808.zip', '20240809.zip', '20240810.zip', '20240811.zip',
        '20240812.zip', '20240813.zip', '20240814.zip', '20240815.zip', '20240816.zip',
        '20240817.zip', '20240818.zip', '20240819.zip', '20240820.zip', '20240821.zip',
        '20240822.zip', '20240823.zip', '20240824.zip', '20240825.zip', '20240826.zip',
        '20240827.zip', '20240828.zip', '20240829.zip', '20240830.zip', '20240831.zip',
        '20240901.zip', '20240902.zip', '20240903.zip', '20240904.zip', '20240905.zip',
        '20240906.zip', '20240907.zip', '20240908.zip', '20240909.zip', '20240910.zip',
        '20240911.zip', '20240912.zip', '20240913.zip', '20240914.zip', '20240915.zip',
        '20240916.zip', '20240917.zip', '20240918.zip', '20240919.zip', '20240920.zip',
        '20240921.zip', '20240922.zip', '20240923.zip', '20240924.zip', '20240925.zip',
        '20240926.zip', '20240927.zip', '20240928.zip', '20240929.zip', '20240930.zip',
        '20241001.zip', '20241002.zip', '20241003.zip', '20241004.zip', '20241005.zip',
        '20241006.zip', '20241007.zip', '20241008.zip', '20241009.zip', '20241010.zip',
        '20241011.zip', '20241012.zip', '20241013.zip', '20241014.zip', '20241015.zip',
        '20241016.zip', '20241017.zip', '20241018.zip', '20241019.zip', '20241020.zip',
        '20241021.zip', '20241022.zip', '20241023.zip', '20241024.zip', '20241025.zip',
        '20241026.zip', '20241027.zip', '20241028.zip', '20241029.zip', '20241030.zip',
        '20241031.zip', '20241101.zip', '20241102.zip', '20241103.zip', '20241104.zip',
        '20241105.zip', '20241106.zip', '20241107.zip', '20241108.zip', '20241109.zip',
        '20241110.zip', '20241111.zip', '20241112.zip', '20241113.zip', '20241114.zip',
        '20241115.zip', '20241116.zip', '20241117.zip', '20241118.zip', '20241119.zip',
        '20241120.zip', '20241121.zip', '20241122.zip', '20241123.zip', '20241124.zip',
        '20241125.zip', '20241126.zip', '20241127.zip', '20241128.zip', '20241129.zip',
        '20241130.zip', '20241201.zip', '20241202.zip', '20241203.zip', '20241204.zip',
        '20241205.zip', '20241206.zip', '20241207.zip', '20241208.zip', '20241209.zip',
        '20241210.zip', '20241211.zip', '20241212.zip', '20241213.zip', '20241214.zip',
        '20241215.zip', '20241216.zip', '20241217.zip', '20241218.zip', '20241219.zip',
        '20241220.zip', '20241221.zip', '20241222.zip', '20241223.zip', '20241224.zip',
        '20241225.zip', '20241226.zip', '20241227.zip', '20241228.zip', '20241229.zip',
        '20241230.zip', '20241231.zip', '20250101.zip', '20250102.zip', '20250103.zip',
        '20250104.zip', '20250105.zip', '20250106.zip', '20250107.zip', '20250108.zip',
        '20250109.zip', '20250110.zip', '20250111.zip', '20250112.zip', '20250113.zip',
        '20250114.zip', '20250115.zip', '20250116.zip', '20250117.zip', '20250118.zip',
        '20250119.zip', '20250120.zip', '20250121.zip', '20250122.zip', '20250123.zip',
        '20250124.zip', '20250125.zip', '20250126.zip', '20250127.zip', '20250128.zip',
        '20250129.zip', '20250130.zip', '20250131.zip', '20250201.zip', '20250202.zip',
        '20250203.zip', '20250204.zip', '20250205.zip', '20250206.zip', '20250207.zip',
        '20250208.zip', '20250209.zip', '20250210.zip', '20250211.zip', '20250212.zip',
        '20250213.zip', '20250214.zip', '20250215.zip', '20250216.zip', '20250217.zip',
        '20250218.zip', '20250219.zip', '20250220.zip', '20250221.zip', '20250222.zip',
        '20250223.zip', '20250224.zip', '20250225.zip', '20250226.zip', '20250227.zip',
        '20250228.zip', '20250301.zip', '20250302.zip', '20250303.zip', '20250304.zip',
        '20250305.zip', '20250306.zip', '20250307.zip', '20250308.zip', '20250309.zip',
        '20250310.zip', '20250311.zip', '20250312.zip', '20250313.zip', '20250314.zip',
        '20250315.zip', '20250316.zip', '20250317.zip', '20250318.zip', '20250319.zip',
        '20250320.zip', '20250321.zip', '20250322.zip', '20250323.zip', '20250324.zip',
        '20250325.zip', '20250326.zip', '20250327.zip', '20250328.zip', '20250329.zip',
        '20250330.zip', '20250331.zip', '20250401.zip', '20250402.zip', '20250403.zip',
        '20250404.zip', '20250405.zip', '20250406.zip', '20250407.zip', '20250408.zip',
        '20250409.zip', '20250410.zip', '20250411.zip', '20250412.zip', '20250413.zip',
        '20250414.zip', '20250415.zip', '20250416.zip', '20250417.zip', '20250418.zip',
        '20250419.zip', '20250420.zip', '20250421.zip', '20250422.zip', '20250423.zip',
        '20250424.zip', '20250425.zip', '20250426.zip', '20250427.zip', '20250428.zip',
        '20250429.zip', '20250430.zip', '20250501.zip', '20250502.zip', '20250503.zip',
        '20250504.zip', '20250505.zip', '20250506.zip', '20250507.zip', '20250508.zip',
        '20250509.zip', '20250510.zip', '20250511.zip', '20250512.zip', '20250513.zip',
        '20250514.zip', '20250515.zip', '20250516.zip', '20250517.zip', '20250518.zip',
        '20250519.zip', '20250520.zip', '20250521.zip', '20250522.zip', '20250523.zip',
        '20250524.zip', '20250525.zip', '20250526.zip', '20250527.zip', '20250528.zip',
        '20250529.zip', '20250530.zip', '20250531.zip', '20250601.zip', '20250602.zip',
        '20250603.zip', '20250604.zip', '20250605.zip', '20250606.zip', '20250607.zip',
        '20250608.zip', '20250609.zip', '20250610.zip', '20250611.zip', '20250612.zip',
        '20250613.zip', '20250614.zip', '20250615.zip', '20250616.zip', '20250617.zip',
        '20250618.zip'
    ]

    # Unimos el nombre de la carpeta con cada nombre de archivo
    zip_files_full_path = [os.path.join(zip_folder, filename) for filename in zip_filenames]

    all_dataframes = []
    print(f"{'='*20} Iniciando el proceso de unión de archivos {'='*20}")

    for zip_path in zip_files_full_path:
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                file_list = z.namelist()
                xlsx_file = next((f for f in file_list if f.endswith('.xlsx')), None)
                
                if xlsx_file:
                    print(f"\n{'*'*10} Procesando archivo XLSX: {xlsx_file} dentro de {zip_path} {'*'*10}")
                    with z.open(xlsx_file) as xlsx_content:
                        df = pd.read_excel(xlsx_content, engine='openpyxl')
                        df_cleaned = clean_crypto_data(df.copy(), xlsx_file)
                        all_dataframes.append(df_cleaned)
                else:
                    print(f"¡Opá! No se encontró un archivo .xlsx dentro de {zip_path}.")
        except FileNotFoundError:
            print(f"¡Opá! No se encontró el archivo ZIP '{zip_path}'. Asegúrate de que la ruta sea correcta.")
        except Exception as e:
            print(f"¡Pucha! Ocurrió un error inesperado al procesar '{zip_path}': {e}")
    
    # Unir todos los DataFrames en uno solo
    if all_dataframes:
        final_df = pd.concat(all_dataframes, ignore_index=True)
        final_df.sort_values('timestamp', inplace=True)
        final_df.drop_duplicates(inplace=True)

        print(f"\n{'='*20} Unión y limpieza completada {'='*20}")
        print("Primeras 5 filas del archivo final unificado:")
        print(final_df.head())
        print("\nInformación final del archivo unificado:")
        final_df.info()

        output_file_name = 'datos_cripto_unificados.csv'
        final_df.to_csv(output_file_name, index=False)
        print(f"\n¡Éxito! El archivo unificado '{output_file_name}' fue guardado correctamente.")
    else:
        print("No se encontraron DataFrames para unir. Revisa las rutas de los archivos.")

if __name__ == "__main__":
    main()