#!/usr/bin/env pdm run python
import subprocess
import os
import time
from datetime import datetime

def ejecutar_script(nombre_archivo):
    if os.path.exists(nombre_archivo):
        print(f"\n--> Ejecutando: {nombre_archivo}...")
        resultado = subprocess.run(["python", nombre_archivo], capture_output=True, text=True)
        
        if resultado.returncode == 0:
            print(f"✅ {nombre_archivo} finalizó con éxito.")
            if resultado.stdout:
                print(resultado.stdout.strip())
        else:
            print(f"❌ Error en {nombre_archivo}:")
            print(resultado.stderr)
            raise RuntimeError(f"El script {nombre_archivo} falló.")
    else:
        raise FileNotFoundError(f"No se encontró el archivo {nombre_archivo}")


def ejecutar_pipeline_completo():
    hoy = datetime.now()
    print("=" * 70)
    print(f"[{hoy:%Y-%m-%d %H:%M:%S}] INICIANDO PIPELINE DE VINCULACIÓN CON EL MEDIO")
    print("=" * 70)
    
    try:
        ejecutar_script("vcm_extraccion.py")
        ejecutar_script("vcm_transformacion.py")
        ejecutar_script("vcm_comentarios.py")
        ejecutar_script("vcm_carga.py")
        
        print("\n" + "=" * 70)
        print(f"[{hoy:%Y-%m-%d %H:%M:%S}] ¡PROCESO COMPLETO FINALIZADO CON ÉXITO!")
        print("=" * 70)
        
    except Exception as e:
        print("\n" + "!" * 70)
        print(f"CRÍTICO - El pipeline se detuvo por un error: {e}")
        print("!" * 70)


def esperar_julio():
    
    ARCHIVO_CONTROL = "ultimo_julio_ejecutado.txt"

    while True:
        ahora = datetime.now()

        # Leer el último año en que se ejecutó en julio
        ultimo_anio = None
        if os.path.exists(ARCHIVO_CONTROL):
            with open(ARCHIVO_CONTROL, "r") as f:
                contenido = f.read().strip()
                if contenido.isdigit():
                    ultimo_anio = int(contenido)

        if ahora.month == 7 and ahora.year != ultimo_anio:
            print(f"\n📅 Es julio de {ahora.year}. Iniciando pipeline...")
            ejecutar_pipeline_completo()

            # Registrar que ya se ejecutó este año
            with open(ARCHIVO_CONTROL, "w") as f:
                f.write(str(ahora.year))

            print(f"\n⏳ Pipeline ejecutado. Próxima ejecución: julio {ahora.year + 1}.")

        else:
            if ahora.month == 7:
                print(f"[{ahora:%Y-%m-%d %H:%M}] Ya se ejecutó en julio {ahora.year}. Esperando julio {ahora.year + 1}...")
            else:
                meses_restantes = (7 - ahora.month) % 12 or 12
                print(f"[{ahora:%Y-%m-%d %H:%M}] Mes actual: {ahora.month}. Faltan ~{meses_restantes} mes(es) para julio. Verificando en 1 hora...")

        # Verificar cada hora
        time.sleep(3600)


if __name__ == "__main__":
    esperar_julio()