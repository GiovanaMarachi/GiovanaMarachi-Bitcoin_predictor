import zipfile
import os

zip_file_path = 'data/20241201.zip'

if not os.path.exists(zip_file_path):
    print(f"Error: El archivo ZIP no se encontró en la ruta: {zip_file_path}")
    print("Asegúrate de que '20241201.zip' esté en la carpeta 'data/'.")
else:
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            print(f"Contenido del archivo ZIP '{zip_file_path}':")
            for name in zf.namelist():
                print(f"- {name}")
    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' no es un archivo ZIP válido o está corrupto.")
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer el ZIP: {e}")