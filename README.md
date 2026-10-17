# 📈 Retail Sales Forecasting - MLOps Pipeline

Este proyecto implementa un modelo de Machine Learning (Random Forest Regressor) para pronosticar las ventas diarias de una cadena minorista (Retail). El proyecto está construido bajo una arquitectura **MLOps**, garantizando la modularidad, reproducibilidad y el seguimiento automatizado de experimentos.

## 🛠️ Tecnologías y Herramientas
* **Python 3**
* **Prefect:** Orquestación del pipeline de datos y entrenamiento.
* **MLflow:** Tracking de experimentos y Model Registry.
* **Docker:** Contenerización de los servidores de Prefect y MLflow.
* **Git:** Control de versiones.

## 📂 Estructura del Proyecto
```text
Actividad-1/
├── configs/
│   └── config.yaml          # Hiperparámetros, rutas y URLs centralizadas
├── data/
│   └── ventas_retail.csv    # Dataset generado
├── flows/
│   └── experimento_flow.py  # Pipeline principal de Prefect
├── notebooks/
│   └── 01_mock_mlflow.ipynb # Pruebas del Model Registry en MLflow
├── src/                     # Código fuente modular
│   ├── config.py            # Lector del archivo YAML
│   ├── datos.py             # Limpieza y preparación de datos
│   ├── generate_data.py     # Script generador de datos sintéticos
│   ├── modelado.py          # Entrenamiento del Random Forest
│   └── tracking.py          # Conexión y registro con MLflow
├── .gitignore
└── README.md
