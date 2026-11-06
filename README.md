# Forecasting de Ventas Retail: arquitectura MLOps

Sistema MLOps para pronosticar las ventas diarias de un retailer con un `RandomForestRegressor`. Automatiza el entrenamiento y la inferencia, sirve el modelo con una API, monitorea el drift de los datos y valida la calidad del modelo antes de desplegarlo.

Proyecto de la Maestría en Ciencia de Datos, Universidad Santo Tomás (MLOps en Python).
**Autores:** Laura Tabares Perez y Harold Steven Rodriguez Cuevas.
**Docente:** Henry Mauricio Cala Castro.

## Tecnologías

| Tecnología | Para qué se usa |
|---|---|
| Scikit-Learn y Pandas | Procesamiento de datos y entrenamiento del Random Forest |
| MLflow | Seguimiento de runs, métricas (RMSE) y registro de modelos con el alias `champion` |
| Prefect | Orquestación de las pipelines y despliegues (entrenamiento manual, inferencia diaria) |
| FastAPI | API que sirve el modelo `champion` |
| SciPy | Prueba de Kolmogorov-Smirnov para detectar drift |
| Pytest | Tests de datos y modelo |

## Cómo funciona

1. **Entrenamiento** (`flows/entrenamiento_flow.py`): ingesta, entrenamiento, evaluación y registro en MLflow. El modelo nuevo recibe el alias `champion` solo si su RMSE es menor al umbral (`registro.umbral_rmse`) y además mejora al champion vigente. Si no, el run queda registrado pero el alias no cambia.
2. **Inferencia** (`flows/inferencia_flow.py`): carga el modelo por el alias `champion` (no por número de versión), predice un lote y guarda las predicciones en `data/predicciones/prediccion_AAAA-MM-DD.csv`. Cada ejecución queda como run en MLflow.
3. **API** (`api/main.py`): descarga el `champion` al iniciar y expone `GET /health` y `POST /predict`. Si cambia el champion, hay que reiniciar la API.
4. **Monitoreo de drift** (`flows/monitoreo_drift_flow.py`): compara la distribución de las ventas de un lote nuevo contra un lote de referencia con la prueba KS. Si el p-value es menor al umbral, emite una alerta y lanza el reentrenamiento.
5. **Quality gate** (`ci_workflow.py`): ejecuta los tests con pytest y bloquea el despliegue si el RMSE del modelo candidato supera el umbral. En la demostración, el RMSE del candidato es un valor simulado (7.500.000).
6. **Equidad** (`scripts/analisis_equidad.py`): calcula el RMSE por zona y verifica que la desviación entre la mejor y la peor no supere el 15%.

Los valores que pueden cambiar (datos, hiperparámetros, MLflow, umbral de RMSE, tamaño del lote) están en `configs/config.yaml`, separados del código. La selección de variables está en `configs/features.yaml`.

## Estructura del repositorio

```
api/         API de FastAPI (main.py)
configs/     configuración en YAML (config.yaml, features.yaml)
flows/       pipelines de Prefect: entrenamiento, inferencia, despliegues y monitoreo de drift
scripts/     carga de datos, reporte de drift y análisis de equidad
src/         lógica reutilizable: configuración, datos, modelado y tracking
tests/       tests con pytest
notebooks/   exploración inicial
```

## Requisitos

- Python 3.10 o superior
- MLflow (puerto 5000) y Prefect (puerto 4200) levantados para entrenar e inferir

## Instalación

```bash
git clone https://github.com/HaroldRC123/Actividad-1.git
cd Actividad-1
pip install -r requirements.txt
```

En cada terminal donde se ejecute un flow o la API, definir las variables de entorno.

En PowerShell:

```powershell
$env:PREFECT_API_URL = "http://localhost:4200/api"
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
```

En Linux o macOS:

```bash
export PREFECT_API_URL="http://localhost:4200/api"
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

## Uso

Desde la raíz del repositorio:

```bash
# Entrenar y, si mejora, promover a champion
python flows/entrenamiento_flow.py

# Inferencia con el champion
python flows/inferencia_flow.py

# Monitoreo de drift (compara los lotes en data/lotes/)
python flows/monitoreo_drift_flow.py

# Quality gate (tests + umbral de RMSE)
python ci_workflow.py

# Registrar los despliegues en Prefect
python flows/desplegar.py
```

### API

Con un modelo `champion` registrado en MLflow:

```bash
uvicorn api.main:app --port 8000
```

Documentación interactiva en http://localhost:8000/docs.

`POST /predict` recibe `dia_semana`, `dia_mes` y `mes`:

```json
{ "dia_semana": 5, "dia_mes": 30, "mes": 10 }
```

y devuelve la predicción de ventas y la versión del modelo:

```json
{ "prediccion_ventas": 7824566.76, "version_modelo": "1" }
```
## Tests

Para ejecutar todos los tests:

```bash
python -m pytest -v
```

Para ejecutar un archivo en particular:

```bash
python -m pytest tests/test_datos.py -v
python -m pytest tests/test_modelo.py -v
python -m pytest tests/test_api.py -v
```

| Archivo | Qué verifica |
|---|---|
| `tests/test_datos.py` | Datos de los lotes en `data/lotes/`: sin nulos y rangos lógicos (cantidades mayores a 0, ventas no negativas) |
| `tests/test_modelo.py` | Lógica del modelo |
| `tests/test_api.py` | Endpoints `/health` y `/predict` de la API, usando un modelo falso (no necesita MLflow) |

## Nota sobre los datos

Los datos del caso son simulados (`src/generate_data.py`). Con ellos, las diferencias de RMSE entre versiones del modelo deben leerse con cautela: el objetivo del proyecto es el mecanismo de MLOps (promoción, despliegue y monitoreo), no la calidad predictiva.
