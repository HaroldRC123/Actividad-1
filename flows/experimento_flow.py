import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prefect import task, flow
import mlflow
import os
import yaml
from datetime import datetime, timedelta

# --- Utilidad para leer configuración ---
def cargar_configuracion(ruta_yaml="configs/config.yaml"):
    with open(ruta_yaml, 'r') as file:
        return yaml.safe_load(file)

# --- Tareas de Prefect ---
@task(name="1. Cargar y Preparar Datos")
def cargar_datos(ruta: str):
    df = pd.read_csv(ruta)
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    ventas_diarias = df.groupby(df['fecha'].dt.date)['ventas'].sum().reset_index()
    ventas_diarias['fecha'] = pd.to_datetime(ventas_diarias['fecha'])
    
    ventas_diarias['dia_semana'] = ventas_diarias['fecha'].dt.dayofweek
    ventas_diarias['dia_mes'] = ventas_diarias['fecha'].dt.day
    ventas_diarias['mes'] = ventas_diarias['fecha'].dt.month
    
    return ventas_diarias.sort_values('fecha')

@task(name="2. Entrenar Modelo de Forecasting")
def entrenar_modelo(df: pd.DataFrame, config_modelo: dict, experiment_name: str):
    X = df[['dia_semana', 'dia_mes', 'mes']]
    y = df['ventas']
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(config_modelo)
        
        modelo = RandomForestRegressor(**config_modelo)
        modelo.fit(X_train, y_train)
        
        predicciones = modelo.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predicciones))
        mae = mean_absolute_error(y_test, predicciones)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.sklearn.log_model(modelo, "modelo_forecasting")
        
        return modelo, rmse, mae

@task(name="3. Resumen y Predicción Futura")
def generar_resumen(modelo, rmse, mae):
    print(f"\n📊 RMSE: ${rmse:,.2f} | MAE: ${mae:,.2f}")
    
    manana = datetime.now() + timedelta(days=1)
    X_nuevo = pd.DataFrame({'dia_semana': [manana.weekday()], 'dia_mes': [manana.day], 'mes': [manana.month]})
    
    prediccion = modelo.predict(X_nuevo)[0]
    print(f"🔮 Predicción para {manana.strftime('%Y-%m-%d')}: ${prediccion:,.2f}\n")

# --- Flow Principal ---
@flow(name="Pipeline Forecasting Modular")
def experimento_flow():
    # 1. Leer YAML
    config = cargar_configuracion()
    
    ruta_datos = config['datos']['ruta_csv']
    if not os.path.exists(ruta_datos):
        raise FileNotFoundError("Genera los datos primero con src/generate_data.py")
    
    # 2. Ejecutar tareas
    df_diario = cargar_datos(ruta_datos)
    modelo, rmse, mae = entrenar_modelo(
        df_diario, 
        config['modelo'], 
        config['mlflow']['experiment_name']
    )
    generar_resumen(modelo, rmse, mae)

if __name__ == "__main__":
    experimento_flow()