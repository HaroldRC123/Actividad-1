import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables de tu modelo de Forecasting
NUMERICAS = ["dia_mes"]
CATEGORICAS = ["dia_semana", "mes"]
VARIABLES = NUMERICAS + CATEGORICAS
OBJETIVO = "ventas" # Cambia a "ventas" si así se llama en tu CSV

def resolver_features(features_cfg):
    """Convierte la selección de features.yaml en dos listas: (numericas, categoricas)."""
    seleccion = features_cfg["seleccion"]
    
    if seleccion == "ALL":
        elegidas = VARIABLES
    elif seleccion == "NUMERIC":
        elegidas = NUMERICAS
    elif seleccion == "CUSTOM":
        elegidas = features_cfg["custom"]
        desconocidas = [v for v in elegidas if v not in VARIABLES]
        if desconocidas:
            raise ValueError(f"Variables desconocidas en 'custom': {desconocidas}")
    else:
        raise ValueError(f"Seleccion '{seleccion}' no válida. Opciones: ALL, NUMERIC, CUSTOM")

    if not elegidas:
        raise ValueError("La selección de features no tiene ninguna variable")
        
    numericas = [v for v in NUMERICAS if v in elegidas]
    categoricas = [v for v in CATEGORICAS if v in elegidas]
    
    return numericas, categoricas

def construir_modelo(tipo, parametros, numericas, categoricas):
    """Preprocesado (escalar y one-hot) más el estimador elegido en la configuración."""
    if tipo == "random_forest_regressor":
        estimador = RandomForestRegressor(**parametros)
    else:
        raise ValueError(f"Modelo '{tipo}' no soportado")

    transformadores = []
    if numericas:
        transformadores.append(("num", StandardScaler(), numericas))
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricas))
        
    preprocesado = ColumnTransformer(transformadores)
    return Pipeline([("preprocesado", preprocesado), ("modelo", estimador)])

def calcular_metricas(y_true, predicciones):
    """Métricas de regresión para evaluar el forecasting."""
    mse = mean_squared_error(y_true, predicciones)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, predicciones)),
        "r2": float(r2_score(y_true, predicciones)),
    }