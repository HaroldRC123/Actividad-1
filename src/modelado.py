import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def entrenar_random_forest(df: pd.DataFrame, config_modelo: dict):
    """Entrena el modelo de regresión y calcula las métricas de error."""
    X = df[['dia_semana', 'dia_mes', 'mes']]
    y = df['ventas']
    
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    modelo = RandomForestRegressor(**config_modelo)
    modelo.fit(X_train, y_train)
    
    predicciones = modelo.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predicciones))
    mae = mean_absolute_error(y_test, predicciones)
    
    return modelo, rmse, mae