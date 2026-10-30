import mlflow
import pytest
import os

def test_modelo_champion_disponible():
    # Solo ejecuta la prueba si la variable de entorno está configurada
    if "MLFLOW_TRACKING_URI" in os.environ:
        try:
            modelo = mlflow.sklearn.load_model("models:/retail-forecasting-modelo@champion")
            assert modelo is not None, "El modelo champion devolvió un objeto vacío."
        except Exception as e:
            pytest.fail(f"❌ Error crítico: No se pudo cargar el modelo de MLflow. {e}")