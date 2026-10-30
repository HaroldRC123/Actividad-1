import subprocess
import yaml
import sys

def run_tests():
    print("🚀 Iniciando Integración Continua (CI)...")
    print("⏳ Ejecutando tests de datos y modelo con Pytest...")
    
    # Ejecuta pytest de forma programática
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"])
    
    if result.returncode != 0:
        print("\n❌ Fallaron los tests técnicos. Pipeline de CI detenido.")
        sys.exit(1)
    print("\n✅ Todos los tests pasaron exitosamente.")

def quality_gate():
    print("\n🚧 Evaluando Quality Gate de Negocio...")
    
    # Leer el umbral máximo tolerado en configuración
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    umbral = config.get("umbral_rmse", 6000000)

    # Simulamos el RMSE de un modelo malo que acaba de terminar su entrenamiento
    rmse_nuevo_modelo = 7500000 

    print(f"📊 RMSE del nuevo modelo candidato: {rmse_nuevo_modelo}")
    print(f"🎯 Umbral máximo permitido (config.yaml): {umbral}")

    if rmse_nuevo_modelo > umbral:
        print("\n❌ QUALITY GATE FALLIDO: El modelo degrada el rendimiento financiero.")
        print("⛔ Acción del CI: Despliegue a producción BLOQUEADO y reportado.")
        sys.exit(1)
    else:
        print("\n✅ QUALITY GATE APROBADO: El modelo cumple con los estándares.")
        print("🚀 Acción del CI: Despliegue autorizado.")

if __name__ == "__main__":
    run_tests()
    quality_gate()