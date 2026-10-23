"""Actividad 2: pipeline de inferencia diaria."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import date
from pathlib import Path
import mlflow
import yaml
from mlflow.tracking import MlflowClient
from prefect import flow, task

# Importamos el NUEVO validador de datos
from scripts.load_data import cargar_datos as leer_datos

CONFIG = "configs/config.yaml"

@task
def cargar_champion(cfg):
    nombre = cfg["registro"]["nombre_modelo"]
    version = MlflowClient().get_model_version_by_alias(nombre, "champion")
    modelo = mlflow.sklearn.load_model(f"models:/{nombre}@champion")
    print(f"Modelo cargado: {nombre} versión {version.version}")
    return modelo, version.version, version.run_id

@task
def ingesta(cfg):
    # ¡Usamos el validador!
    df = leer_datos(cfg["datos"]["ruta"])
    return df.sample(n=cfg["inferencia"]["tamano_lote"])

@task
def predecir(modelo, lote, cfg):
    predicciones = modelo.predict(lote[list(modelo.feature_names_in_)])
    
    resultado = lote.copy()
    resultado["prediccion_ventas"] = predicciones

    carpeta = Path(cfg["inferencia"]["carpeta_salida"])
    carpeta.mkdir(parents=True, exist_ok=True)
    ruta = carpeta / f"prediccion_{date.today().isoformat()}.csv"
    resultado.to_csv(ruta, index=False)
    
    print(f"Predicciones guardadas en {ruta}")
    return str(ruta), len(predicciones), float(predicciones.mean())

@task
def registrar_inferencia(version, run_id_entrenamiento, ruta, filas, venta_media):
    with mlflow.start_run(run_name=f"inferencia-{date.today().isoformat()}"):
        mlflow.set_tag("fase", "inferencia")
        mlflow.set_tag("modelo_version", version)
        mlflow.set_tag("modelo_run_id", run_id_entrenamiento)
        mlflow.log_metric("filas_predichas", filas)
        mlflow.log_metric("venta_media_predicha", venta_media)
        mlflow.log_artifact(ruta)

@flow(name="inferencia-forecasting", log_prints=True)
def inferencia_flow():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    
    modelo, version, run_id_entrenamiento = cargar_champion(cfg)
    lote = ingesta(cfg)
    ruta, filas, venta_media = predecir(modelo, lote, cfg)
    registrar_inferencia(version, run_id_entrenamiento, ruta, filas, venta_media)

if __name__ == "__main__":
    inferencia_flow()