import pandas as pd
import numpy as np
import os

def generar_datos_logistica(n_samples: int = 1000, random_seed: int = 42, output_path: str = "data/entregas_logistica.csv"):
    """
    Genera un conjunto de datos dummy para predicción de retrasos en entregas.

    Parámetros:
    - n_samples (int): Número de registros a generar.
    - random_seed (int): Semilla para garantizar la reproducibilidad del experimento.
    - output_path (str): Ruta donde se guardará el archivo CSV generado.
    """
    # Fijar la semilla para que los datos sean reproducibles
    np.random.seed(random_seed)

    # Generar características (features)
    distancia_km = np.random.uniform(5, 1200, n_samples)
    peso_kg = np.random.uniform(1, 500, n_samples)
    clima_adverso = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]) # 20% de probabilidad de lluvia/tormenta
    trafico_pesado = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4]) # 40% de probabilidad de trancón

    # Generar variable objetivo (target) 'retraso' basada en reglas lógicas
    # A mayor distancia, clima adverso o tráfico, mayor es la probabilidad de que el envío se retrase
    prob_retraso = (distancia_km / 1200 * 0.3) + (clima_adverso * 0.4) + (trafico_pesado * 0.3)
    prob_retraso = np.clip(prob_retraso, 0, 1) # Asegurar que la probabilidad esté entre 0 y 1

    # 1 = Retrasado, 0 = A tiempo
    retraso = np.random.binomial(1, p=prob_retraso)

    # Crear el DataFrame
    df = pd.DataFrame({
        "distancia_km": distancia_km.round(2),
        "peso_kg": peso_kg.round(2),
        "clima_adverso": clima_adverso,
        "trafico_pesado": trafico_pesado,
        "retraso": retraso
    })

    # Crear la carpeta contenedora si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Exportar a CSV
    df.to_csv(output_path, index=False)
    print(f"✅ Se generaron {n_samples} registros con semilla {random_seed}.")
    print(f"📂 Archivo guardado en: {output_path}")

if __name__ == "__main__":
    # Ejecución por defecto con los parámetros requeridos
    generar_datos_logistica(n_samples=2000, random_seed=123)