"""Actividad 2: despliegue de las pipelines en Prefect."""
from prefect import serve

from entrenamiento_flow import entrenamiento_flow
from inferencia_flow import inferencia_flow

if __name__ == "__main__":
    despliegue_entrenamiento = entrenamiento_flow.to_deployment(name="entrenamiento-retail")
    # cron "0 6 * * *" significa todos los días a las 6:00 AM
    despliegue_inferencia = inferencia_flow.to_deployment(name="inferencia-diaria-retail", cron="0 6 * * *")
    
    print("Iniciando servidor de despliegues en Prefect...")
    serve(despliegue_entrenamiento, despliegue_inferencia)