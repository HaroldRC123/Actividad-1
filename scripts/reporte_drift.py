import pandas as pd
from scipy.stats import ks_2samp
import os

def generar_reporte_drift(ruta_base: str, rutas_lotes: list, features_a_monitorear: list):
    """
    Compara estadísticamente un dataset base contra varios lotes nuevos para detectar data drift.
    """
    print(f"📊 Iniciando Análisis de Data Drift...")
    print(f"Base de referencia (Training): {ruta_base}\n" + "-"*50)
    
    df_base = pd.read_csv(ruta_base)
    
    for ruta_lote in rutas_lotes:
        if not os.path.exists(ruta_lote):
            continue
            
        df_lote = pd.read_csv(ruta_lote)
        print(f"\n🔍 Analizando Lote: {ruta_lote}")
        
        reporte = []
        for col in features_a_monitorear:
            # El test de Kolmogorov-Smirnov compara dos distribuciones.
            # Un p-value < 0.05 indica que las distribuciones son estadísticamente diferentes (hay drift).
            stat, p_value = ks_2samp(df_base[col], df_lote[col])
            drift_detectado = p_value < 0.05
            
            reporte.append({
                "Variable": col,
                "Drift": "⚠️ SÍ" if drift_detectado else "✅ NO",
                "Media Entrenamiento": round(df_base[col].mean(), 2),
                "Media Lote": round(df_lote[col].mean(), 2),
                "p-value": round(p_value, 5)
            })
            
        df_reporte = pd.DataFrame(reporte)
        print(df_reporte.to_string(index=False))
        print("-" * 50)

if __name__ == "__main__":
    # Usamos el lote de drift 0.0 como nuestra "foto original" de cómo era el negocio
    base = "data/lotes/ventas_lote_drift_0.0.csv"
    
    # Lotes nuevos que llegan en los periodos siguientes
    lotes_nuevos = [
        "data/lotes/ventas_lote_drift_0.3.csv",
        "data/lotes/ventas_lote_drift_0.6.csv"
    ]
    
    # Variables donde queremos vigilar si el comportamiento cambió
    columnas_clave = ["ventas", "cantidad", "costo_total"]
    
    generar_reporte_drift(base, lotes_nuevos, columnas_clave)