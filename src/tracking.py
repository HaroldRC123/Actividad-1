import mlflow

def registrar_experimento(experiment_name: str, tracking_uri: str, config_modelo: dict, modelo, rmse: float, mae: float):
    """Registra los parámetros, métricas y el artefacto del modelo en MLflow remoto."""
    
    # 1. Apuntar al servidor de Docker
    mlflow.set_tracking_uri(tracking_uri)
    
    # 2. Configurar experimento y registrar
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_params(config_modelo)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        
        mlflow.sklearn.log_model(
            modelo, 
            "modelo_forecasting",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )