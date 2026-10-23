import pandas as pd
import yaml
from src.datos import cargar_y_preparar_datos  # Importamos tu preparador de datos

RUTA_CONFIG = "configs/config.yaml"

COLUMNAS = [
    "dia_semana",
    "dia_mes",
    "mes",
    "ventas"  # Ajustado al nombre real de tu variable a predecir
]

def cargar_datos(ruta=None):
    """Devuelve el DataFrame preparado con las columnas esperadas."""
    if ruta is None:
        with open(RUTA_CONFIG, encoding="utf-8") as f:
            ruta = yaml.safe_load(f)["datos"]["ruta"]

    # 1. Cargamos y PREPARAMOS los datos usando tu código original
    df = cargar_y_preparar_datos(ruta)
    
    # 2. El guardia de seguridad verifica
    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Error de Datos: Faltan columnas: {faltan}. Columnas actuales: {list(df.columns)}")
        
    return df[COLUMNAS].copy()

def main():
    df = cargar_datos()
    print(f"✅ Datos validados. Filas: {len(df)} | Columnas: {df.shape[1]}")

if __name__ == "__main__":
    main()