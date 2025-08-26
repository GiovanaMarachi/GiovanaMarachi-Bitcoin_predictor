# bitcoin_predictor/walk_forward.py
import os, math, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Semillas para reproducibilidad básica
import random
seed = 7
np.random.seed(seed); tf.random.set_seed(seed); random.seed(seed); os.environ["PYTHONHASHSEED"]=str(seed)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs" / "walk_forward"

# Dataset y parámetros en modo SMOKE
SMOKE = os.getenv("SMOKE_TEST", "0") == "1"
if SMOKE:
    DATA_DIR = BASE_DIR / "tests" / "datasets"
    EXCEL_NAME = "small.xlsx"
    seq_length = 10
    min_train = 80
    horizon = 20
    step = 20
    epochs = 5
    batch_size = 16
else:
    EXCEL_NAME = "20241201.xlsx"
    seq_length = 20
    min_train = 600            # ajusta según tu dataset real
    horizon = 100              # tamaño del bloque de test
    step = 100                 # avance entre ventanas
    epochs = 25
    batch_size = 64

excel_file_path = DATA_DIR / EXCEL_NAME
assert excel_file_path.exists(), f"No se encuentra el archivo: {excel_file_path}"

# Alias para columnas
alias_map = {
    "open": ["open", "apertura"],
    "high": ["high", "max", "alto"],
    "low":  ["low", "min", "bajo"],
    "close": ["close", "cierre", "closeprice", "precio_cierre"],
    "basevolume": ["basevolume", "volume", "volumen_base"],
    "usdtvolume": ["usdtvolume", "quotevolume", "volumen_usdt", "volumen_quote"]
}

def pick(df, target):
    for name in alias_map[target]:
        if name in df.columns: return name
    raise ValueError(f"Falta una columna equivalente a '{target}'. Columnas: {list(df.columns)}")

def create_sequences_range(scaled, seq_len, close_idx, start_t, end_t):
    # crea X,y para tiempos t en [start_t, end_t)
    X, y = [], []
    s = max(seq_len, start_t)
    for t in range(s, end_t):
        X.append(scaled[t-seq_len:t, :])
        y.append(scaled[t, close_idx])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# Cargar datos
df = pd.read_excel(excel_file_path, engine="openpyxl")
df.columns = [str(c).strip().lower() for c in df.columns]

open_col  = pick(df, "open")
high_col  = pick(df, "high")
low_col   = pick(df, "low")
close_col = pick(df, "close")
basev_col = pick(df, "basevolume")
usdtv_col = pick(df, "usdtvolume")

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["timestamp"]).set_index("timestamp")

for c in [open_col, high_col, low_col, close_col, basev_col, usdtv_col]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=[open_col, high_col, low_col, close_col, basev_col, usdtv_col])

features = [open_col, high_col, low_col, close_col, basev_col, usdtv_col]
data = df[features].values
n = len(data)
assert n >= (min_train + horizon + seq_length), \
    f"Datos insuficientes: {n}. Requiere al menos {min_train + horizon + seq_length}"

run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
RUN_DIR = OUT_DIR / f"run_{run_id}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

rows = []
close_idx = features.index(close_col)
eps = 1e-8
start_train_end = max(min_train, seq_length + 1)
i = start_train_end

print(f"Iniciando walk-forward. Total filas: {n}, min_train: {min_train}, horizon: {horizon}, step: {step}")

while i + horizon <= n:
    train_end = i
    test_end = i + horizon

    train_data = data[:train_end]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train_data)

    all_until_test = data[:test_end]
    scaled_all = scaler.transform(all_until_test)

    # Conjuntos
    X_train, y_train = create_sequences_range(
        scaled_all, seq_length, close_idx, start_t=seq_length, end_t=train_end
    )
    X_test, y_test = create_sequences_range(
        scaled_all, seq_length, close_idx, start_t=train_end, end_t=test_end
    )

    if len(X_train) == 0 or len(X_test) == 0:
        i += step
        continue

    model = build_model((X_train.shape[1], X_train.shape[2]))
    callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                     validation_data=(X_test, y_test), verbose=0, callbacks=callbacks)

    # Predicción en test
    y_pred_norm = model.predict(X_test, verbose=0).reshape(-1)

    # Desescalar a USD
    dummy_pred = np.zeros((len(y_pred_norm), len(features))); dummy_pred[:, close_idx] = y_pred_norm
    pred_usd = scaler.inverse_transform(dummy_pred)[:, close_idx]

    dummy_real = np.zeros((len(y_test), len(features))); dummy_real[:, close_idx] = y_test
    real_usd = scaler.inverse_transform(dummy_real)[:, close_idx]

    # Métricas normalizadas
    mae_n  = mean_absolute_error(y_test, y_pred_norm)
    rmse_n = np.sqrt(mean_squared_error(y_test, y_pred_norm))
    mape_n = np.mean(np.abs((y_test - y_pred_norm) / (np.abs(y_test) + eps))) * 100
    prec_n = 100 - mape_n

    # Métricas USD
    mae_u  = mean_absolute_error(real_usd, pred_usd)
    rmse_u = np.sqrt(mean_squared_error(real_usd, pred_usd))
    mape_u = np.mean(np.abs((real_usd - pred_usd) / (np.abs(real_usd) + eps))) * 100
    prec_u = 100 - mape_u

    row = {
        "train_end_idx": train_end,
        "test_start_idx": train_end,
        "test_end_idx": test_end,
        "n_train_seq": len(X_train),
        "n_test_seq": len(X_test),
        "MAE_norm": mae_n, "RMSE_norm": rmse_n, "MAPE_norm (%)": mape_n, "Precision_norm (%)": prec_n,
        "MAE_usd": mae_u, "RMSE_usd": rmse_u, "MAPE_usd (%)": mape_u, "Precision_usd (%)": prec_u
    }
    rows.append(row)
    print(f"Ventana train:[0,{train_end}) test:[{train_end},{test_end})  MAE_usd={mae_u:.4f}  MAPE_usd={mape_u:.4f}%")

    i += step

# Guardar resumen
res = pd.DataFrame(rows)
res.to_csv(RUN_DIR / "walk_forward_summary.csv", index=False)

# Promedios globales
agg = {
    "MAE_norm": res["MAE_norm"].mean(),
    "RMSE_norm": res["RMSE_norm"].mean(),
    "MAPE_norm (%)": res["MAPE_norm (%)"].mean(),
    "Precision_norm (%)": res["Precision_norm (%)"].mean(),
    "MAE_usd": res["MAE_usd"].mean(),
    "RMSE_usd": res["RMSE_usd"].mean(),
    "MAPE_usd (%)": res["MAPE_usd (%)"].mean(),
    "Precision_usd (%)": res["Precision_usd (%)"].mean(),
    "windows": len(res)
}
pd.DataFrame([agg]).to_csv(RUN_DIR / "walk_forward_agg.csv", index=False)

print("Walk-forward terminado.")
print(f"Carpeta: {RUN_DIR}")
