"""Actividad 2: pipeline de entrenamiento para Forecasting."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import mlflow
import yaml
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from prefect import flow, task
from sklearn.model_selection import train_test_split

# Importamos tu lógica de negocio y el NUEVO validador de datos
from src.modelado import OBJETIVO, calcular_metricas, construir_modelo, resolver_features
from scripts.load_data import cargar_datos as leer_datos

CONFIG = "configs/config.yaml"
CONFIG_FEATURES = "configs/features.yaml"

@task
def cargar_datos(ruta):
    # ¡Aquí usamos el validador en lugar de pd.read_csv!
    return leer_datos(ruta)

@task
def entrenar_y_registrar(df, cfg, features_cfg):
    numericas, categoricas = resolver_features(features_cfg)
    variables = numericas + categoricas
    division = cfg["division"]
    
    train_df, test_df = train_test_split(
        df,
        test_size=division["test_size"],
        random_state=division["random_state"],
        shuffle=False
    )
    X_train, y_train = train_df[variables], train_df[OBJETIVO]
    X_test, y_test = test_df[variables], test_df[OBJETIVO]

    tipo = cfg["modelo"]["tipo"]
    parametros = cfg["modelo"].get("parametros") or {}
    modelo = construir_modelo(tipo, parametros, numericas, categoricas)
    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    metricas = calcular_metricas(y_test, predicciones)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    mlflow.set_experiment(cfg["mlflow"]["experimento"])
    
    with mlflow.start_run(run_name=cfg["mlflow"]["nombre_run"]) as run:
        mlflow.set_tag("fase", "entrenamiento")
        mlflow.log_params({
            "modelo": tipo,
            **parametros,
            "semilla": division["random_state"],
            "features": features_cfg["seleccion"],
            "n_features": len(variables),
        })
        mlflow.log_metrics(metricas)
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            signature=infer_signature(X_test, predicciones),
            input_example=X_test.head(3),
        )
        run_id = run.info.run_id

    return run_id, info.model_uri, metricas

@task
def promover_si_mejora(model_uri, metricas, cfg):
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"]))
    client = MlflowClient()
    nombre = cfg["registro"]["nombre_modelo"]
    rmse = metricas["rmse"]

    if rmse > cfg["registro"]["umbral_rmse"]:
        print(f"RMSE {rmse:.2f} supera el máximo permitido de {cfg['registro']['umbral_rmse']}: no se registra como champion")
        return None

    try:
        champion = client.get_model_version_by_alias(nombre, "champion")
    except MlflowException:
        champion = None 

    if champion is not None:
        rmse_champion = client.get_run(champion.run_id).data.metrics["rmse"]
        if rmse >= rmse_champion:
            print(f"RMSE {rmse:.2f} no mejora al champion ({rmse_champion:.2f}): el alias no cambia")
            return None

    version = mlflow.register_model(model_uri, nombre)
    client.set_registered_model_alias(nombre, "champion", version.version)
    client.set_model_version_tag(nombre, version.version, "estado", "aprobado")
    print(f"Versión {version.version} registrada como champion (RMSE {rmse:.2f})")
    return version.version

@flow(name="entrenamiento-forecasting", log_prints=True)
def entrenamiento_flow():
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(CONFIG_FEATURES, encoding="utf-8") as f:
        features_cfg = yaml.safe_load(f)

    df = cargar_datos(cfg["datos"]["ruta"])
    run_id, model_uri, metricas = entrenar_y_registrar(df, cfg, features_cfg)
    promover_si_mejora(model_uri, metricas, cfg)
    print(f"Run de MLflow: {run_id} | RMSE = {metricas['rmse']:.2f}")
    return run_id

if __name__ == "__main__":
    entrenamiento_flow()