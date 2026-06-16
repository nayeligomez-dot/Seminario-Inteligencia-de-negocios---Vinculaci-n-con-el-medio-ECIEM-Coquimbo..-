import os
import json
import logging
from typing import List, Dict, Any
 
import requests
import pandas as pd
 
# ─── Configuración ────────────────────────────────────────────────────────────
API_URL  = "https://www.eciem.cl/api/api_datos.php"
TOKEN    = "Eciem_20252026"
TABLAS   = ["alumn_pract", "alumn_pract_eva_inf_pract", "alumn_pract_eva_jef"]
LIMIT    = 2000
DATA_DIR = "data"
 
os.makedirs(DATA_DIR, exist_ok=True)
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
log = logging.getLogger(__name__)
 
 
# ─── Funciones ────────────────────────────────────────────────────────────────
 
def _get_json(url: str, params: Dict[str, Any], timeout: int = 40) -> Dict:
    """GET con manejo de errores HTTP y de parseo JSON."""
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        raise RuntimeError(f"Respuesta no es JSON: {r.text[:300]}")
 
 
def descargar_tabla(tabla: str) -> pd.DataFrame:
    """
    Descarga todas las páginas de una tabla desde la API.
    Estructura esperada del JSON:
        { "tabla": "...", "total": N, "filas": [ {...}, {...} ] }
    """
    log.info("[EXTRACCION] Descargando tabla: %s", tabla)
    offset    = 0
    pagina    = 1
    acumulado: List[Dict] = []
 
    while True:
        log.info("  → página %d (offset=%d)", pagina, offset)
        payload = _get_json(API_URL, {
            "tabla": tabla, "token": TOKEN,
            "limit": LIMIT, "offset": offset,
        })
 
        if "filas" not in payload:
            raise RuntimeError(f"Respuesta sin 'filas' para {tabla}: {json.dumps(payload)[:400]}")
 
        filas  = payload["filas"]
        total  = int(payload.get("total", 0))
        acumulado.extend(filas)
        log.info("  → %d filas recibidas (total esperado: %d)", len(filas), total)
 
        if not filas or offset + LIMIT >= total:
            break
 
        offset += LIMIT
        pagina += 1
 
    df = pd.DataFrame(acumulado)
    log.info("[EXTRACCION] %s: %d filas descargadas, %d columnas", tabla, len(df), len(df.columns))
    return df
 
 
def limpiar_basico(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima: strip de strings, fechas inválidas a NaT."""
    if df.empty:
        return df
 
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().replace({"nan": None, "None": None})
 
    fechas = [c for c in df.columns if c.lower().startswith(("fech_", "fecha_"))]
    for col in fechas:
        df[col] = df[col].replace({"0000-00-00": None, "0000-00-00 00:00:00": None})
        df[col] = pd.to_datetime(df[col], errors="coerce")
 
    return df
 
 
def guardar_csv(df: pd.DataFrame, nombre: str) -> str:
    ruta = os.path.join(DATA_DIR, f"{nombre}.csv")
    df.to_csv(ruta, index=False, encoding="utf-8")
    log.info("[EXTRACCION] Guardado: %s (%d filas)", ruta, len(df))
    return ruta
 
 
# ─── Main ─────────────────────────────────────────────────────────────────────
 
def main():
    if not TOKEN or TOKEN.startswith("pon"):
        raise ValueError("Configura el TOKEN antes de ejecutar.")
 
    for tabla in TABLAS:
        df = descargar_tabla(tabla)
        df = limpiar_basico(df)
        guardar_csv(df, tabla)
 
    log.info("[EXTRACCION] Extracción completada ✔")
 
 
if __name__ == "__main__":
    main()
 