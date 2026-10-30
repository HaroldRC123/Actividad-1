import pandas as pd
import mlflow
import numpy as np
from sklearn.metrics import mean_squared_error
import os

def evaluar_equidad():
    print("⚖️ Iniciando Análisis de Equidad del Modelo (Fairness)...")
    
    # 1. Cargar datos de prueba
    ruta_datos = "data/lotes/ventas_lote_drift_0.0.csv"
    df = pd.read_csv(ruta_datos)
    
    # 2. Conectar a MLflow y cargar el modelo en producción
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000"
    try:
        modelo = mlflow.sklearn.load_model("models:/retail-forecasting-modelo@champion")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
        
    # 3. Preparar variables predictoras 
    # (Ajusta estas columnas según lo que tengas configurado en tu features.yaml)
    X = df[['dia_semana', 'dia_mes', 'mes', 'cantidad', 'costo_total']]
    y_real = df['ventas']
    
    # 4. Generar predicciones
    df['prediccion_ventas'] = modelo.predict(X)
    
    # 5. Calcular RMSE General
    rmse_general = np.sqrt(mean_squared_error(y_real, df['prediccion_ventas']))
    print(f"\n📊 RMSE General del Lote: {rmse_general:,.2f}\n")
    
    # 6. Calcular métricas por Subgrupo (Sucursal)
    print("🔍 Rendimiento segmentado por Sucursal:")
    resultados = []
    
    for sucursal in df['sucursal'].unique():
        df_sub = df[df['sucursal'] == sucursal]
        rmse_sub = np.sqrt(mean_squared_error(df_sub['ventas'], df_sub['prediccion_ventas']))
        diferencia = rmse_sub - rmse_general
        
        resultados.append({
            "Sucursal": sucursal,
            "RMSE": round(rmse_sub, 2),
            "Desviación vs General": round(diferencia, 2)
        })
        
    df_resultados = pd.DataFrame(resultados).sort_values(by="RMSE")
    print(df_resultados.to_string(index=False))
    
    # 7. Documentar diferencias (Tolerancia del 15% de desviación)
    print("\n📝 Documentación de Equidad:")
    max_desviacion = abs(df_resultados['Desviación vs General']).max()
    
    if max_desviacion > (rmse_general * 0.15):
        print("⚠️ ALERTA: Se detectaron sesgos. El modelo favorece ciertas sucursales perjudicando a otras.")
    else:
        print("✅ APROBADO: El rendimiento es equitativo. No hay diferencias estadísticamente perjudiciales entre sucursales.")

if __name__ == "__main__":
    evaluar_equidad()