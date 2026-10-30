import pandas as pd
import pytest

@pytest.fixture
def datos():
    return pd.read_csv("data/lotes/ventas_lote_drift_0.0.csv")

def test_sin_nulos(datos):
    assert datos.isnull().sum().sum() == 0, "❌ Error: Hay valores nulos en el dataset de ingesta."

def test_rangos_logicos(datos):
    assert (datos['cantidad'] > 0).all(), "❌ Error: Hay cantidades negativas o cero en los tickets."
    assert (datos['ventas'] >= 0).all(), "❌ Error: Hay ventas con valores financieros negativos."