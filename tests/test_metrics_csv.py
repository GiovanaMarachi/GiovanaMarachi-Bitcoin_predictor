import os, sys, subprocess, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "lstm_btc_predictor.py"
OUT = ROOT / "outputs"

def run_smoke():
    env = os.environ.copy()
    env["SMOKE_TEST"] = "1"
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT), env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

def test_metrics_csv_has_both_scales_and_numbers():
    run_smoke()
    runs = sorted([p for p in OUT.glob("run_*") if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = runs[0]
    metrics = pd.read_csv(run_dir / "metrics.csv")

    # Columnas esperadas
    expected_cols = {
        "MAE_norm","RMSE_norm","MAPE_norm (%)","Precisión_norm (%)",
        "MAE_usd","RMSE_usd","MAPE_usd (%)","Precisión_usd (%)"
    }
    assert expected_cols.issubset(set(metrics.columns)), f"Faltan columnas: {expected_cols - set(metrics.columns)}"

    row = metrics.iloc[0]
    # Valores numéricos válidos
    for c in expected_cols:
        val = float(row[c])
        assert val == val and val != float("inf") and val >= 0, f"Valor inválido en {c}: {val}"

    # Rango razonable (no obligatorio, pero útil)
    assert 0 <= row["MAPE_norm (%)"] <= 100
    assert 0 <= row["MAPE_usd (%)"] <= 100
    assert 0 <= row["Precisión_norm (%)"] <= 100
    assert 0 <= row["Precisión_usd (%)"] <= 100
