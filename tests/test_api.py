"""Tests de la API con un modelo falso (sin MLflow)."""
import sys
from types import SimpleNamespace

import mlflow.sklearn
import mlflow.tracking
import numpy as np
import pytest
from fastapi.testclient import TestClient


class ModeloFalso:
    feature_names_in_ = np.array(["dia_semana", "dia_mes", "mes"])

    def predict(self, X):
        return np.array([1234567.891])


@pytest.fixture
def client(monkeypatch):
    # api.main fija MLFLOW_TRACKING_URI al importarse; monkeypatch lo restaura al terminar
    # para no afectar a otros tests (como test_modelo.py).
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setattr(mlflow.sklearn, "load_model", lambda uri: ModeloFalso())
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", lambda: SimpleNamespace(
        get_model_version_by_alias=lambda nombre, alias: SimpleNamespace(version="7")))
    sys.modules.pop("api.main", None)
    import api.main as main
    return TestClient(main.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["version"] == "7"


def test_predict(client):
    r = client.post("/predict", json={"dia_semana": 5, "dia_mes": 30, "mes": 10})
    assert r.status_code == 200
    assert r.json() == {"prediccion_ventas": 1234567.89, "version_modelo": "7"}
