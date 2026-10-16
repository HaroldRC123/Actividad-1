import yaml

def cargar_configuracion(ruta_yaml="configs/config.yaml"):
    """Carga los parámetros desde el archivo YAML."""
    with open(ruta_yaml, 'r') as file:
        return yaml.safe_load(file)