import pandas as pd
from scipy.stats import ks_2samp
from prefect import flow, task
import yaml
import subprocess
import sys

@task
def cargar_umbral_drift(ruta_config="configs/config.yaml"): # Ajusta la ruta si tu yaml está en otro lado
    try:
        with open(ruta_config, "r") as f:
            config = yaml.safe_load(f)
            return config.get("drift_threshold_p_value", 0.05)
    except FileNotFoundError:
        return 0.05

@task
def evaluar_drift(ruta_base, ruta_nuevo_lote, umbral_p_value):
    df_base = pd.read_csv(ruta_base)
    df_lote = pd.read_csv(ruta_nuevo_lote)
    
    # Evaluamos drift solo en la variable objetivo 'ventas'
    stat, p_value = ks_2samp(df_base['ventas'], df_lote['ventas'])
    drift_detectado = p_value < umbral_p_value
    
    print(f"📊 P-value de ventas: {p_value:.5f} (Umbral de tolerancia: {umbral_p_value})")
    return drift_detectado

@flow(name="monitoreo-drift-reentrenamiento")
def monitoreo_drift_flow():
    umbral = cargar_umbral_drift()
    
    # Simulamos la ingesta del lote original y la llegada del lote alterado
    ruta_base = "data/lotes/ventas_lote_drift_0.0.csv"
    ruta_lote_nuevo = "data/lotes/ventas_lote_drift_0.6.csv" 
    
    hay_drift = evaluar_drift(ruta_base, ruta_lote_nuevo, umbral)
    
    if hay_drift:
        print("🚨 ¡Alerta! Drift detectado en la distribución de ventas. Iniciando reentrenamiento automático...")
        # Llama a tu pipeline de entrenamiento de la Actividad 2
        subprocess.run([sys.executable, "flows/entrenamiento_flow.py"])
        print("🔄 Reentrenamiento y registro completado.")
    else:
        print("✅ Comportamiento de datos estable. No es necesario reentrenar.")

if __name__ == "__main__":
    monitoreo_drift_flow()