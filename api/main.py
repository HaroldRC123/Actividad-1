import os
import mlflow
import pandas as pd
import yaml
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

# 1. Cargar configuración
with open("configs/config.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
NOMBRE_MODELO = cfg["registro"]["nombre_modelo"]

# 2. Conectar a MLflow y cargar el modelo CHAMPION
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
modelo = mlflow.sklearn.load_model(f"models:/{NOMBRE_MODELO}@champion")
VERSION = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO, "champion").version

# 3. Inicializar FastAPI
app = FastAPI(
    title="API de Forecasting de Ventas", 
    description="Predicción de volumen de ventas diarias utilizando el modelo champion."
)

# 4. Definir la estructura de los datos de entrada
class VentaDia(BaseModel):
    """Datos temporales para realizar la predicción de ventas de un día."""
    dia_semana: int
    dia_mes: int
    mes: int
    # Si en tu features.yaml agregaste más columnas numéricas o categóricas, ponlas aquí.

@app.get("/health")
def health():
    """Endpoint para comprobar que la API está viva y qué versión del modelo usa."""
    return {"estado": "ok", "modelo": NOMBRE_MODELO, "version": VERSION}

@app.post("/predict")
def predict(venta: VentaDia):
    """Endpoint principal para realizar una predicción."""
    # Convertir la entrada a un DataFrame de pandas
    datos = pd.DataFrame([venta.model_dump()])
    
    # Asegurar que las columnas estén en el mismo orden que cuando se entrenó
    datos = datos[list(modelo.feature_names_in_)]
    
    # Calcular la predicción (regresión directa, no probabilidad)
    prediccion = float(modelo.predict(datos)[0])
    
    return {
        "prediccion_ventas": round(prediccion, 2),
        "version_modelo": VERSION,
    }