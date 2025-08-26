import os, re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "lstm_btc_predictor.py"
OUT = ROOT / "outputs"

def test_e2e_smoke_runs_and_writes_artifacts():
    env = os.environ.copy()
    env["SMOKE_TEST"] = "1"

    # Ejecuta el script
    proc = subprocess.run([sys.executable, str(SCRIPT)], cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout)
    print(proc.stderr)
    assert proc.returncode == 0, "El script falló en modo prueba."

    # Encuentra la última carpeta run_*
    runs = [p for p in OUT.glob("run_*") if p.is_dir()]
    assert runs, "No se creó ninguna carpeta run_* en outputs/"
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    run_dir = runs[0]

    # Verificar archivos mínimos
    required = {"model.h5","prediction.png","loss.png","history.csv","metrics.csv"}
    names = {p.name for p in run_dir.iterdir()}
    assert required.issubset(names), f"Faltan artefactos en {run_dir}: {required - names}"

    # Formato de nombre de carpeta
    assert re.match(r"run_\d{8}-\d{6}", run_dir.name), "Nombre de run_* no cumple formato."
