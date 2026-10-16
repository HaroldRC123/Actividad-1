import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from prefect import task, flow
import mlflow
import os

@task(name="Cargar Datos")
def cargar_datos(ruta: str):
    return pd.read_csv(ruta)

@task(name="Entrenar Modelo")
def entrenar_modelo(df: pd.DataFrame):
    X = df.drop("retraso", axis=1)
    y = df["retraso"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Prediccion_Retrasos")
    with mlflow.start_run():
        parametros = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
        mlflow.log_params(parametros)
        
        modelo = RandomForestClassifier(**parametros)
        modelo.fit(X_train, y_train)
        
        precision = accuracy_score(y_test, modelo.predict(X_test))
        mlflow.log_metric("accuracy", precision)
        mlflow.sklearn.log_model(modelo, "modelo_rf")
        
        print(f"✅ Modelo entrenado. Accuracy: {precision:.4f}")
        return modelo

@flow(name="Pipeline Principal")
def main_flow():
    ruta = "data/entregas_logistica.csv"
    if not os.path.exists(ruta):
        raise FileNotFoundError("Ejecuta src/generate_data.py primero.")
    entrenar_modelo(cargar_datos(ruta))

if __name__ == "__main__":
    main_flow()