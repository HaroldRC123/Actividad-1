import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generar_datos_retail(n_samples: int = 1000, random_seed: int = 42, output_path: str = "data/ventas_retail.csv"):
    """
    Genera un conjunto de datos dummy de ventas de retail para modelos de Machine Learning.
    """
    np.random.seed(random_seed)

    # 1. Generar Fechas aleatorias de este año (Enero a Octubre 2026)
    start_date = datetime(2026, 1, 1)
    end_date = datetime(2026, 10, 15)
    dias_totales = (end_date - start_date).days
    dias_aleatorios = np.random.randint(0, dias_totales, n_samples)
    fechas = [start_date + timedelta(days=int(d)) for d in dias_aleatorios]

    # 2. Generar Dimensiones (Sucursales, SKUs y Tickets)
    sucursales = np.random.choice(['Sucursal_Norte', 'Sucursal_Centro', 'Sucursal_Sur', 'Sucursal_Occidente'], n_samples)
    skus = np.random.choice([f'SKU_{i}' for i in range(1001, 1050)], n_samples)
    tickets = [f'TXN-{np.random.randint(100000, 999999)}' for _ in range(n_samples)]

    # 3. Generar Métricas Financieras (Cantidades, Costos y Ventas)
    cantidad = np.random.randint(1, 25, n_samples)
    costo_unitario = np.random.uniform(5000, 150000, n_samples)
    
    # Aplicar un margen de ganancia aleatorio (entre 10% y 70% sobre el costo)
    markup = np.random.uniform(1.1, 1.7, n_samples) 
    precio_unitario = costo_unitario * markup

    costo_total = costo_unitario * cantidad
    ventas = precio_unitario * cantidad

    # Crear DataFrame
    df = pd.DataFrame({
        "fecha": fechas,
        "ticket": tickets,
        "sucursal": sucursales,
        "sku": skus,
        "cantidad": cantidad,
        "costo_total": costo_total.round(2),
        "ventas": ventas.round(2)
    })

    # 4. Variable Objetivo (Target) para Clasificación
    # 1 = Margen Alto (Ganancia superior al 40%), 0 = Margen Normal/Bajo
    df['margen_alto'] = ((df['ventas'] - df['costo_total']) > (df['costo_total'] * 0.4)).astype(int)

    # Crear la carpeta contenedora si no existe y guardar
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Se generaron {n_samples} registros de retail con semilla {random_seed}.")
    print(f"📂 Archivo guardado en: {output_path}")

if __name__ == "__main__":
    generar_datos_retail(n_samples=2000, random_seed=123)