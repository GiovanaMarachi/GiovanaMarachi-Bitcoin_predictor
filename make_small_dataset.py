from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / "tests" / "datasets"
BASE.mkdir(parents=True, exist_ok=True)
path = BASE / "small.xlsx"

# 120 filas, columnas requeridas por tu script
n = 120
np.random.seed(7)

close = np.cumsum(np.random.randn(n) * 5 + 0.2) + 30000  # serie “tipo precio”
high  = close + np.abs(np.random.randn(n) * 2)
low   = close - np.abs(np.random.randn(n) * 2)
open_ = close + np.random.randn(n)
basevolume = np.abs(np.random.randn(n) * 10_000) + 1_000
usdtvolume = basevolume * (close / np.mean(close))

df = pd.DataFrame({
    "open": open_,
    "high": high,
    "low": low,
    "close": close,
    "basevolume": basevolume,
    "usdtvolume": usdtvolume
})
df.to_excel(path, index=False)
print(f"Creado: {path}")
