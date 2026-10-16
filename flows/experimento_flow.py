import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from datetime import datetime, timedelta
from prefect import task, flow

# Importar la lógica desde la carpeta src
from src.config import cargar_configuracion
from src.datos import cargar_y_preparar_datos
from src.modelado import entrenar_random_forest
from src.tracking import registrar_experimento

@task(name="1. Cargar y Preparar Datos")
def task_cargar_datos(ruta: str):
    return cargar_y_preparar_datos(ruta)

@task(name="2. Entrenar Modelo y Registrar en MLflow")
def task_entrenar_y_registrar(df: pd.DataFrame, config_modelo: dict, config_mlflow: dict):
    modelo, rmse, mae = entrenar_random_forest(df, config_modelo)
    
    # Pasamos el experiment_name y el tracking_uri
    registrar_experimento(
        config_mlflow['experiment_name'], 
        config_mlflow['tracking_uri'], 
        config_modelo, 
        modelo, 
        rmse, 
        mae
    )
    return modelo, rmse, mae

@task(name="3. Resumen y Predicción Futura")
def task_generar_resumen(modelo, rmse, mae):
    print(f"\n📊 RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f}")
    
    manana = datetime.now() + timedelta(days=1)
    X_nuevo = pd.DataFrame({'dia_semana': [manana.weekday()], 'dia_mes': [manana.day], 'mes': [manana.month]})
    
    prediccion = modelo.predict(X_nuevo)[0]
    print(f"🔮 Predicción para {manana.strftime('%Y-%m-%d')}: ${prediccion:,.2f}\n")

@flow(name="Pipeline Forecasting Modular")
def experimento_flow():
    config = cargar_configuracion()
    ruta_datos = config['datos']['ruta_csv']
    
    if not os.path.exists(ruta_datos):
        raise FileNotFoundError("Genera los datos primero ejecutando: python src/generate_data.py")
    
    df_diario = task_cargar_datos(ruta_datos)
    
    modelo, rmse, mae = task_entrenar_y_registrar(
        df_diario, 
        config['modelo'], 
        config['mlflow']
    )
    
    task_generar_resumen(modelo, rmse, mae)
    modelo, rmse, mae = task_entrenar_y_registrar(
        df_diario, 
        config['modelo'], 
        config['mlflow']  # Pasamos el bloque completo de mlflow del YAML
    )
if __name__ == "__main__":
    experimento_flow()