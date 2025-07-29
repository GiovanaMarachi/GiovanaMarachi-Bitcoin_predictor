import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import zipfile
import io

# Paso 1: Cargar datos reales de Bitcoin desde el ZIP (ahora es un archivo XLSX)
zip_file_path = 'data/20241201.zip'
# Nombre del archivo XLSX *dentro* del archivo ZIP
excel_file_in_zip = '20241201.xlsx' # ¡Cambiado a .xlsx!

# Abrir el archivo ZIP y leer el XLSX específico
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zf:
        with zf.open(excel_file_in_zip) as f:
            # Usar pd.read_excel en lugar de pd.read_csv
            # Se asume que los datos están en la primera hoja (Sheet1) por defecto
            df = pd.read_excel(f)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo ZIP en la ruta: {zip_file_path}")
    print("Asegúrate de que '20241201.zip' esté en la carpeta 'data/'.")
    exit()
except KeyError:
    print(f"Error: No se encontró el archivo '{excel_file_in_zip}' dentro del ZIP.")
    print(f"Asegúrate de que el nombre del archivo dentro del ZIP sea exactamente '{excel_file_in_zip}'.")
    exit()
except Exception as e:
    print(f"Ocurrió un error al leer el archivo Excel: {e}")
    print("Asegúrate de que tienes la librería 'openpyxl' instalada para leer archivos .xlsx:")
    print("pip install openpyxl")
    exit()

# Resto del código sigue igual, asumiendo que las columnas son las mismas
# Convertir la columna 'timestamp' a formato de fecha y hora
# Los timestamps parecen estar en segundos, así que multiplicamos por 10^9 para nanosegundos para pandas
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# Establecer 'timestamp' como índice (opcional, pero buena práctica para series de tiempo)
df = df.set_index('timestamp')

# Seleccionar las características relevantes (OHLCV)
features = ['open', 'high', 'low', 'close', 'basevolume', 'usdtvolume']
data = df[features].values

# Paso 2: Normalizar las características seleccionadas
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Paso 3: Crear secuencias para múltiples características
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        # X contendrá las últimas 'seq_len' filas con todas las características
        X.append(data[i - seq_len:i])
        # y contendrá el precio de cierre del día actual (última columna de 'data' antes de escalar)
        # Asumiendo que 'close' es la columna en el índice 3 de 'features' list
        y.append(data[i, features.index('close')])
    return np.array(X), np.array(y)

# Longitud de la secuencia (número de pasos de tiempo pasados para predecir el siguiente)
seq_length = 20

X, y = create_sequences(scaled_data, seq_length)

print("Forma de X (secuencias, timesteps, features):", X.shape)
print("Forma de y (salida):", y.shape)

# Paso 4: Dividir en entrenamiento y test (división basada en el tiempo)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Paso 5: Crear modelo LSTM
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

# Paso 6: Compilar y entrenar
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

# Paso 7: Guardar modelo entrenado
if not os.path.exists('models'):
    os.makedirs('models')
model.save('models/lstm_model_bitcoin_real_data.h5')

# Paso 8: Predicción
predicted = model.predict(X_test)

dummy_array_for_inverse = np.zeros((len(predicted), len(features)))
dummy_array_for_inverse[:, features.index('close')] = predicted.flatten()

predicted_prices = scaler.inverse_transform(dummy_array_for_inverse)[:, features.index('close')]

dummy_array_for_inverse_y_test = np.zeros((len(y_test), len(features)))
dummy_array_for_inverse_y_test[:, features.index('close')] = y_test.flatten()

real_prices = scaler.inverse_transform(dummy_array_for_inverse_y_test)[:, features.index('close')]

# Paso 9: Visualización
plt.figure(figsize=(12, 6))
plt.plot(real_prices, label='Precio Real')
plt.plot(predicted_prices, label='Precio Predicho')
plt.title('Predicción de Precios de Bitcoin con Datos Reales (Diciembre 2024)')
plt.xlabel('Puntos de Datos')
plt.ylabel('Precio (USD)')
plt.legend()
plt.grid(True)

if not os.path.exists('outputs'):
    os.makedirs('outputs')
plt.savefig('outputs/bitcoin_prediction_real_data_plot.png')

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de Validación')
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('outputs/training_loss_plot.png')