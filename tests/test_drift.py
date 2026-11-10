"""Tests de la detección de drift. No necesitan MLflow ni Prefect levantados."""
import numpy as np
import pandas as pd

from flows.monitoreo_drift_flow import cargar_umbral_drift, evaluar_drift


def _guardar(tmp_path, nombre, ventas):
    ruta = tmp_path / nombre
    pd.DataFrame({"ventas": ventas}).to_csv(ruta, index=False)
    return str(ruta)


def test_no_hay_drift_con_la_misma_distribucion(tmp_path):
    rng = np.random.default_rng(0)
    base = _guardar(tmp_path, "base.csv", rng.normal(1000, 100, 500))
    lote = _guardar(tmp_path, "lote.csv", rng.normal(1000, 100, 500))
    assert not evaluar_drift.fn(base, lote, 0.05)


def test_detecta_drift_si_las_ventas_suben(tmp_path):
    rng = np.random.default_rng(0)
    base = _guardar(tmp_path, "base.csv", rng.normal(1000, 100, 500))
    lote = _guardar(tmp_path, "lote.csv", rng.normal(1600, 100, 500))
    assert evaluar_drift.fn(base, lote, 0.05)


def test_umbral_por_defecto_si_no_existe_el_config():
    assert cargar_umbral_drift.fn("no_existe.yaml") == 0.05
