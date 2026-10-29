# Forecasting de Ventas Retail - Arquitectura MLOps

Este repositorio contiene la implementación de una arquitectura MLOps completa para la predicción de ventas en el sector retail. El proyecto abarca la automatización del ciclo de vida del modelo predictivo (Random Forest) garantizando trazabilidad, despliegue continuo y disponibilización de inferencias en tiempo real y batch.

Proyecto desarrollado como parte de la Maestría en Ciencia de Datos de la Universidad Santo Tomás (USTA).

## 🏗️ Arquitectura del Sistema

La solución está construida sobre cuatro pilares tecnológicos:
- **Scikit-Learn & Pandas:** Procesamiento de datos y entrenamiento del modelo `RandomForestRegressor`.
- **MLflow:** Model Registry y Tracking. Gestión de métricas (RMSE) y asignación condicional de la etiqueta `@champion` para pases a producción.
- **Prefect:** Orquestación de flujos de trabajo (pipelines). Despliegues automatizados para entrenamiento bajo demanda e inferencia programada.
- **FastAPI:** Capa de servicio orientada a microservicios. API RESTful que consume el modelo campeón en tiempo real.

## ⚙️ Requisitos Previos

- Python 3.10+
- Servidores locales de MLflow (puerto 5000) y Prefect (puerto 4200) inicializados.

## 🚀 Instalación y Configuración

1. **Clonar el repositorio y acceder al directorio:**
   ```powershell
   git clone <URL_DEL_REPOSITORIO>
   cd Actividad-2# Forecasting de Ventas Retail - Arquitectura MLOps

Este repositorio contiene la implementación de una arquitectura MLOps completa para la predicción de ventas en el sector retail. El proyecto abarca la automatización del ciclo de vida del modelo predictivo (Random Forest) garantizando trazabilidad, despliegue continuo y disponibilización de inferencias en tiempo real y batch.

Proyecto desarrollado como parte de la Maestría en Ciencia de Datos de la Universidad Santo Tomás (USTA).

## 🏗️ Arquitectura del Sistema

La solución está construida sobre cuatro pilares tecnológicos:
- **Scikit-Learn & Pandas:** Procesamiento de datos y entrenamiento del modelo `RandomForestRegressor`.
- **MLflow:** Model Registry y Tracking. Gestión de métricas (RMSE) y asignación condicional de la etiqueta `@champion` para pases a producción.
- **Prefect:** Orquestación de flujos de trabajo (pipelines). Despliegues automatizados para entrenamiento bajo demanda e inferencia programada.
- **FastAPI:** Capa de servicio orientada a microservicios. API RESTful que consume el modelo campeón en tiempo real.

## ⚙️ Requisitos Previos

- Python 3.10+
- Servidores locales de MLflow (puerto 5000) y Prefect (puerto 4200) inicializados.

## 🚀 Instalación y Configuración

1. **Clonar el repositorio y acceder al directorio:**
   ```powershell
   git clone <URL_DEL_REPOSITORIO>
   cd Actividad-2