import numpy as np
import pandas as pd
import os  
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from datetime import datetime

# ================== Rutas relativas al archivo actual ==================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
EXCEL_NAME = "20241201.xlsx"

# --- Modo prueba (activar con SMOKE_TEST=1) ---
SMOKE = os.getenv("SMOKE_TEST", "0") == "1"
if SMOKE:
    # Dataset y parámetros mini para tests
    DATA_DIR = BASE_DIR / "tests" / "datasets"
    EXCEL_NAME = "small.xlsx"
    # forzar hiperparámetros pequeños para que el test sea rápido
    seq_length = 10
    epochs = 3
    batch_size = 16
else:
    # valores “normales” de producción/experimentos
    seq_length = 20
    epochs = 50
    batch_size = 64

excel_file_path = DATA_DIR / EXCEL_NAME


# ======= Crear carpeta de corrida (run_YYYYMMDD-HHMMSS) =======
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = OUT_DIR / f"run_{run_id}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

# ================== Cargar Excel ==================
if not excel_file_path.exists():
    print(f" No se encontró el Excel en: {excel_file_path}")
    print("   (Asegúrate de que esté en bitcoin_predictor/data/)")
    sys.exit(1)

try:
    # pip install openpyxl
    df = pd.read_excel(excel_file_path, engine="openpyxl")
    print(f" Excel leído: {excel_file_path}")
except Exception as e:
    print(f" Error al leer el Excel: {e}")
    print("   Instala el lector:  pip install openpyxl")
    sys.exit(1)

# ================== Normalización de columnas ==================
df.columns = [str(c).strip().lower() for c in df.columns]

# Alias típicos
alias_map = {
    "open": ["open", "apertura"],
    "high": ["high", "max", "alto"],
    "low":  ["low", "min", "bajo"],
    "close": ["close", "cierre", "closeprice", "precio_cierre"],
    "basevolume": ["basevolume", "volume", "volumen_base"],
    "usdtvolume": ["usdtvolume", "quotevolume", "volumen_usdt", "volumen_quote"]
}

def pick(target):
    for name in alias_map[target]:
        if name in df.columns:
            return name
    raise ValueError(
        f"Falta una columna equivalente a '{target}'. "
        f"Columnas encontradas: {list(df.columns)}"
    )

open_col  = pick("open")
high_col  = pick("high")
low_col   = pick("low")
close_col = pick("close")
basev_col = pick("basevolume")
usdtv_col = pick("usdtvolume")

# Timestamp opcional
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")

# Asegurar numéricos y limpiar
for c in [open_col, high_col, low_col, close_col, basev_col, usdtv_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[open_col, high_col, low_col, close_col, basev_col, usdtv_col])

features = [open_col, high_col, low_col, close_col, basev_col, usdtv_col]
data = df[features].values

# ================== Parámetros ==================
seq_length = 20
epochs = 50
batch_size = 64

if len(data) < seq_length + 1:
    print(f" Muy pocos registros ({len(data)}). Se necesitan al menos {seq_length+1}.")
    sys.exit(1)

# ================== Escalado ==================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ================== Secuencias ==================
def create_sequences(data, seq_len, close_idx):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i, close_idx])
    return np.array(X), np.array(y)

close_idx = features.index(close_col)
X, y = create_sequences(scaled_data, seq_length, close_idx)

print("Forma de X (secuencias, timesteps, features):", X.shape)
print("Forma de y (salida):", y.shape)

# ================== Split ==================
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ================== Modelo ==================
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")

history = model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_test, y_test),
    verbose=1
)

# ================== Guardar modelo ==================
model_path = RUN_DIR / "model.h5"
model.save(model_path)

# ================== Predicción y desescalado ==================
predicted = model.predict(X_test)                 # (n_test, 1) en escala 0-1 (columna close)

# Desescalar close para predicción y verdad
dummy_pred = np.zeros((len(predicted), len(features)))
dummy_pred[:, close_idx] = predicted.flatten()
predicted_prices = scaler.inverse_transform(dummy_pred)[:, close_idx]

dummy_real = np.zeros((len(y_test), len(features)))
dummy_real[:, close_idx] = y_test.flatten()
real_prices = scaler.inverse_transform(dummy_real)[:, close_idx]

# ================== MÉTRICAS (normalizado y USD) ==================
eps = 1e-8
# 1) Normalizado (0-1)
y_true_norm = y_test.reshape(-1)
y_pred_norm = predicted.reshape(-1)
mae_norm  = mean_absolute_error(y_true_norm, y_pred_norm)
rmse_norm = np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))
mape_norm = np.mean(np.abs((y_true_norm - y_pred_norm) / (np.abs(y_true_norm) + eps))) * 100
precision_norm = 100 - mape_norm

#  USD(desescalado)
y_true_usd = real_prices.reshape(-1)
y_pred_usd = predicted_prices.reshape(-1)
mae_usd  = mean_absolute_error(y_true_usd, y_pred_usd)
rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
mape_usd = np.mean(np.abs((y_true_usd - y_pred_usd) / (np.abs(y_true_usd) + eps))) * 100
precision_usd = 100 - mape_usd
# Guardar CSV de métricas
metrics_path = RUN_DIR / "metrics.csv"
pd.DataFrame({
    "MAE_norm": [mae_norm],
    "RMSE_norm": [rmse_norm],
    "MAPE_norm (%)": [mape_norm],
    "Precisión_norm (%)": [precision_norm],
    "MAE_usd": [mae_usd],
    "RMSE_usd": [rmse_usd],
    "MAPE_usd (%)": [mape_usd],
    "Precisión_usd (%)": [precision_usd],
}).to_csv(metrics_path, index=False)
print("\n Métricas:")
print(f"   [Norm] MAE={mae_norm:.6f}  RMSE={rmse_norm:.6f}  MAPE={mape_norm:.4f}%  Precisión={precision_norm:.4f}%")
print(f"   [USD ] MAE={mae_usd:.4f}   RMSE={rmse_usd:.4f}   MAPE={mape_usd:.4f}%   Precisión={precision_usd:.4f}%")
print(f"   Guardadas en: {metrics_path}")

# ================== Gráficos (guardados en RUN_DIR) ==================
pred_plot = RUN_DIR / "prediction.png"
loss_plot = RUN_DIR / "loss.png"

plt.figure(figsize=(12, 6))
plt.plot(y_true_usd, label="Precio Real (USD)")
plt.plot(y_pred_usd, label="Precio Predicho (USD)")
plt.title("Predicción de Precios de Bitcoin (Datos Reales)")
plt.xlabel("Puntos de Datos")
plt.ylabel("Precio (USD)")
plt.legend()
plt.grid(True)
plt.savefig(pred_plot)
plt.close()

plt.figure(figsize=(12, 6))
plt.plot(history.history["loss"], label="Pérdida Entrenamiento")
plt.plot(history.history["val_loss"], label="Pérdida Validación")
plt.title("Pérdida del Modelo durante el Entrenamiento")
plt.xlabel("Época")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.savefig(loss_plot)
plt.close()

# ================== Historial a CSV ==================
hist_csv = RUN_DIR / "history.csv"
pd.DataFrame(history.history).to_csv(hist_csv, index=False)

print("\n Proceso completado.")
print(f"   Carpeta de corrida: {RUN_DIR}")
print(f"   Modelo:   {model_path}")
print(f"   Gráficos: {pred_plot}")
print(f"             {loss_plot}")
print(f"   History:  {hist_csv}")
print(f"   Métricas: {metrics_path}")
