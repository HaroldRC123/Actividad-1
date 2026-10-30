import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generar_lote_retail(n_samples: int = 1000, random_seed: int = 42, drift: float = 0.0, output_path: str = "data/ventas_retail.csv"):
    """
    Genera un conjunto de datos dummy de ventas de retail con un nivel de drift controlado.
    """
    np.random.seed(random_seed)

    # 1. Generar Fechas aleatorias de este año
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 10, 15)
    dias_totales = (end_date - start_date).days
    dias_aleatorios = np.random.randint(0, dias_totales, n_samples)
    fechas = [start_date + timedelta(days=int(d)) for d in dias_aleatorios]

    # 2. Generar Dimensiones
    sucursales = np.random.choice(['Sucursal_Norte', 'Sucursal_Centro', 'Sucursal_Sur', 'Sucursal_Occidente'], n_samples)
    skus = np.random.choice([f'SKU_{i}' for i in range(1001, 1050)], n_samples)
    tickets = [f'TXN-{np.random.randint(100000, 999999)}' for _ in range(n_samples)]

    # 3. Generar Métricas base
    cantidad = np.random.randint(1, 25, n_samples)
    costo_unitario = np.random.uniform(5000, 150000, n_samples)
    
    markup = np.random.uniform(1.1, 1.7, n_samples) 
    precio_unitario = costo_unitario * markup

    costo_total = costo_unitario * cantidad
    ventas_base = precio_unitario * cantidad

    # 4. INYECCIÓN DE DRIFT (Concept Drift)
    # Si drift = 0.3, las ventas suben un 30% artificialmente. 
    # Esto simulará un cambio drástico en el comportamiento del negocio.
    ventas_con_drift = ventas_base * (1 + drift)

    # Crear DataFrame
    df = pd.DataFrame({
        "fecha": fechas,
        "ticket": tickets,
        "sucursal": sucursales,
        "sku": skus,
        "cantidad": cantidad,
        "costo_total": costo_total.round(2),
        "ventas": ventas_con_drift.round(2)
    })

    # Extraer variables temporales para el modelo de forecasting
    df['fecha'] = pd.to_datetime(df['fecha'])
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia_mes'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month

    # Crear la carpeta contenedora si no existe y guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Lote generado: {n_samples} registros | Drift: {drift} | Archivo guardado en: {output_path}")

if __name__ == "__main__":
    # Generar los 3 lotes de datos solicitados por la rúbrica
    niveles_drift = [0.0, 0.3, 0.6]
    
    for drift_val in niveles_drift:
        generar_lote_retail(
            n_samples=2000, 
            random_seed=123 + int(drift_val * 10), # Variamos la semilla para que los lotes tengan datos distintos
            drift=drift_val,
            output_path=f"data/lotes/ventas_lote_drift_{drift_val}.csv"
        )