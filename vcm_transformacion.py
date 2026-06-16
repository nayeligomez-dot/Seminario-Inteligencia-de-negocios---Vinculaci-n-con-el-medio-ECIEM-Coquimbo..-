import os
import re
import logging
from typing import Optional

import pandas as pd

DATA_DIR   = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "transformado")
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

SIN_INFO = "Sin información"

# ── Normalización de nombres de empresa (usada en dim_empresa Y fact_hechos) ──

_ALIASES_EMPRESA = {}  # se puebla en construir_dim_empresa la primera vez


def _normalizar_nombre(s: str) -> str:
    """
    Normaliza el nombre de empresa aplicando aliases y limpiando puntuación/tildes.
    Usada tanto en construir_dim_empresa como en construir_fact_hechos para
    garantizar que ambas funciones usen el mismo nombre canónico.
    """
    import re as _re
    if not isinstance(s, str):
        return s
    s_title = s.strip().title()
    if s_title in _ALIASES_EMPRESA:
        return _ALIASES_EMPRESA[s_title]
    for orig, rep in zip("áéíóúÁÉÍÓÚäëïöüÄËÏÖÜñÑ", "aeiouAEIOUaeiouAEIOUnn"):
        s_title = s_title.replace(orig, rep)
    s_title = _re.sub(r"[/\\\-,]", " ", s_title)
    s_title = _re.sub(r"\.", "", s_title)
    s_title = _re.sub(r"\s+", " ", s_title).strip()
    return s_title.title()



COLS_ITEMS = ["r_2_1", "r_2_2", "r_2_3", "r_2_4", "r_2_5",
              "r_2_6", "r_2_8", "r_2_9", "r_2_10"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_fecha(valor) -> Optional[pd.Timestamp]:
    if pd.isna(valor) or str(valor).strip() in ("", "0000-00-00", "0000-00-00 00:00:00", "None", "nan"):
        return None
    try:
        return pd.to_datetime(valor, errors="coerce")
    except Exception:
        return None


def _normalizar_rut(rut: str) -> str:
    """Elimina separadores y retorna dígitos+K en mayúscula."""
    if not rut:
        return ""
    return re.sub(r"[^0-9kK]", "", str(rut)).upper()


def _rut_cuerpo(rut: str) -> str:
    """
    Retorna solo el cuerpo numérico del RUT (sin dígito verificador),
    para comparaciones tolerantes cuando el DV está pegado o ausente.
    Ejemplo: '170160657' → '17016065'
             '17016065'  → '1701606'
    """
    limpio = re.sub(r"[^0-9]", "", str(rut))
    return limpio[:-1] if len(limpio) >= 2 else limpio


def _normalizar_carrera(valor: str) -> str:
    v = str(valor).strip().lower()
    for c in "áéíóú":
        v = v.replace(c, "aeiou"["áéíóú".index(c)])
    if any(p in v for p in ["comercial", "ingenmieria", "ingecoucn"]):
        return "Ingeniería Comercial"
    if any(p in v for p in ["informacion", "control", "gestion", "icg", "ing. en"]):
        return "Ingeniería en Información y Control de Gestión"
    if "comercial" in v:
        return "Ingeniería Comercial"
    return SIN_INFO


def _normalizar_nivel_estudio(valor) -> Optional[int]:
    """
    Normaliza el nivel de estudio a número de semestre.
    Retorna 8, 9, 10 o None si no reconoce el valor.
    """
    if not valor or str(valor).strip().lower() in ("", "nan", "none"):
        return None
    v = str(valor).strip().lower().replace("°", "").replace("º", "")
    mapa = {
        "octavo": 8,  "8": 8,  "8vo": 8,  "viii": 8,
        "noveno": 9,  "9": 9,  "9no": 9,  "ix":   9,
        "decimo": 10, "10": 10, "10mo": 10, "x":   10,
        "sexto":  6,  "6": 6,  "6to": 6,  "vi":   6,
        "septimo": 7, "7": 7,  "7mo": 7,  "vii":  7,
    }
    return mapa.get(v)


def _calcular_periodicidad(fecha) -> Optional[str]:
    """
    Clasifica la práctica como verano o invierno según el mes de inicio.
    Verano: diciembre, enero, febrero, marzo (meses 12, 1, 2, 3)
    Invierno: resto del año
    """
    if fecha is None:
        return None
    try:
        ts = pd.Timestamp(fecha)
        return "verano" if ts.month in (12, 1, 2, 3) else "invierno"
    except Exception:
        return None


def _guardar(df: pd.DataFrame, nombre: str):
    ruta = os.path.join(OUTPUT_DIR, f"{nombre}.csv")
    df.to_csv(ruta, index=False, encoding="utf-8")
    log.info("[TRANSFORM] %s: %d filas → %s", nombre, len(df), ruta)


def _leer_raw(nombre: str) -> pd.DataFrame:
    ruta = os.path.join(DATA_DIR, f"{nombre}.csv")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encontrado: {ruta}. Ejecuta vcm_extraccion.py primero.")
    return pd.read_csv(ruta, low_memory=False)


# ─── Dimensiones ──────────────────────────────────────────────────────────────

def construir_dim_rubro(df_pract: pd.DataFrame) -> pd.DataFrame:
    rubros = (
        df_pract["act_econo_empresa"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.title()
        .unique()
    )
    rubros = [r for r in rubros if r.lower() not in ("nan", "none", "")]
    rubros = sorted(set(rubros)) + [SIN_INFO]
    df = pd.DataFrame({"id_rubro": range(1, len(rubros) + 1), "descripcion_rubro": rubros})
    _guardar(df, "dim_rubro")
    return df


def construir_dim_empresa(df_pract: pd.DataFrame, df_rubro: pd.DataFrame) -> pd.DataFrame:
    mapa_rubro  = dict(zip(df_rubro["descripcion_rubro"], df_rubro["id_rubro"]))
    id_sin_info = mapa_rubro.get(SIN_INFO, 1)

    # ── Diccionario de aliases ────────────────────────────────────────────────
    # Mapea variantes de escritura al nombre canónico.
    # Solo incluye casos donde es SEGURO que son la misma empresa.
    # Formato: "variante exacta (title case)" → "nombre canónico"
    _ALIASES_EMPRESA_LOCAL = {
        # ── Banco Estado ──────────────────────────────────────────────────────
        "Bancoestado":                                  "Banco Estado",
        "Banco Estado De Chile":                        "Banco Estado",
        "Banco Del Estado":                             "Banco Estado",
        "Banco Estado Microempresas S.A":               "Banco Estado",
        # ── Banco Santander ───────────────────────────────────────────────────
        "Banco Santander":                              "Banco Santander Chile",
        "Santander":                                    "Banco Santander Chile",
        # ── BCI ───────────────────────────────────────────────────────────────
        "Banco Bci":                                    "Banco Credito E Inversiones",
        "Banco De Créditos E Inversiones":              "Banco Credito E Inversiones",
        "Banco De Credito E Inversiones":               "Banco Credito E Inversiones",
        "Banco De Crédito E Inversiones Bci":           "Banco Credito E Inversiones",
        "Banco De Crédito E Inversiones":               "Banco Credito E Inversiones",
        # ── Banco Chile ───────────────────────────────────────────────────────
        "Banco De Chile":                               "Banco Chile",
        # ── BBVA ──────────────────────────────────────────────────────────────
        "Banco Bilbao Vizcaya Argentaria Chile":        "Banco Bbva",
        # ── Itaú ──────────────────────────────────────────────────────────────
        "Banco Itau Corpbanca":                         "Banco Itau",
        "Itaú Corpbanca":                               "Banco Itau",
        # ── Transbank ─────────────────────────────────────────────────────────
        "Transbank S.A.":                               "Transbank",
        # ── CMP ───────────────────────────────────────────────────────────────
        "Compañia Minera Del Pacífico":                 "Compañia Minera Del Pacifico S.A.",
        "Compañia Minera Del Pacifica":                 "Compañia Minera Del Pacifico S.A.",
        "Compañia Minera Del Pacifico S.A":             "Compañia Minera Del Pacifico S.A.",
        "Compañia Minera Del Pacífico S.A.":            "Compañia Minera Del Pacifico S.A.",
        "Compañia Minera Del Pacífico S.A":             "Compañia Minera Del Pacifico S.A.",
        "Compañía Minera Del Pacífico S.A.":            "Compañia Minera Del Pacifico S.A.",
        "Compañía Minera Del Pacífico S.A":             "Compañia Minera Del Pacifico S.A.",
        "Compañía Minera Del Pacifico S.A":             "Compañia Minera Del Pacifico S.A.",
        "Compañía Minera Del Pacifico":                 "Compañia Minera Del Pacifico S.A.",
        "Compañía Minera Del Pacífico.":                "Compañia Minera Del Pacifico S.A.",
        "Compañía Minera Del Pacífico":                 "Compañia Minera Del Pacifico S.A.",
        "Compañia Minera Del Pacifico S.A - Cmp":       "Compañia Minera Del Pacifico S.A.",
        "Compania Minera Del Pacifico":                 "Compañia Minera Del Pacifico S.A.",
        "Compania Minera Del Pacifico S.A":             "Compañia Minera Del Pacifico S.A.",
        "Compañia Minera El Pacífico":                  "Compañia Minera Del Pacifico S.A.",
        # ── Ecomac (todas las filiales → Ecomac) ──────────────────────────────
        "Ecomac Empresas S.A.":                         "Ecomac",
        "Ecomac Empresas":                              "Ecomac",
        "Inmobiliaria Ecomac S.A.":                     "Ecomac",
        "Inmobiliaria Ecomac S.A":                      "Ecomac",
        "Inmobiliaria  Ecomac S.A":                     "Ecomac",
        "Servicios Inmobiliarios Ecomac S.A.":          "Ecomac",
        "Servicios Inmobiliarios Ecomac S.A":           "Ecomac",
        # ── NYS Clínicas ──────────────────────────────────────────────────────
        "Nys Clinicas":                                 "Nys Clínicas",
        "N&S Clinicas":                                 "Nys Clínicas",
        "N&S Clínicas":                                 "Nys Clínicas",
        # ── UCN ───────────────────────────────────────────────────────────────
        "Universidad Catolica Del Norte":               "Universidad Católica Del Norte",
        "Ucn":                                          "Universidad Católica Del Norte",
        "Universidad Catolica Del Norte Programa Pace": "Universidad Católica Del Norte",
        "Universidad Catolica Del Norte - Facultad De Medicina":   "Universidad Católica Del Norte",
        "Universidad Catolica Del Norte / Facultad De Medicina":   "Universidad Católica Del Norte",
        "Universidad Católica Del Norte - Unidad De Éxito Académico": "Universidad Católica Del Norte",
        "Universidad Católica Del Norte - Escuela De Ciencias Empresariales": "Universidad Católica Del Norte",
        "Universidad Catolica Del Norte - Deptp. De Comunicaciones": "Universidad Católica Del Norte",
        # ── Codelco (divisiones separadas, solo corregir escritura) ───────────
        "Codelco Divisiãƒâ€Œn Andina":                 "Codelco Division Andina",
        "Corporación Nacional Del Cobre De Chile":      "Codelco",
        "Codelco Chile Division Salvador":              "Codelco Division Salvador",
        # ── ENAMI ─────────────────────────────────────────────────────────────
        "Empresa Nacional De Mineria - Enami":          "Empresa Nacional De Minería",
        "Empresa Nacional De Mineria":                  "Empresa Nacional De Minería",
        # ── CCU ───────────────────────────────────────────────────────────────
        "Compañía De Cervecerías Unidas":               "Ccu",
        "Comercial Ccu S.A.":                           "Ccu",
        "Compañía De Cerveceras Unidas Ccu":            "Ccu",
        "Compañía Cervecería Unidas S.A. Ccu":          "Ccu",
        # ── CORFO ─────────────────────────────────────────────────────────────
        "Corfo":                                        "Corporación De Fomento De La Producción",
        "Corporacion De Fomento De La Produccion":      "Corporación De Fomento De La Producción",
        "Corporacion De Fomento De La Producción - Corfo": "Corporación De Fomento De La Producción",
        "Corporacion De Fomento De La Produccion - Corfo":  "Corporación De Fomento De La Producción",
        "Corporación De Fomento De La Producción Corfo Dr Coquimbo": "Corporación De Fomento De La Producción",
        "Corporacin De Fomento De La Produccion":       "Corporación De Fomento De La Producción",
        "Corfo Dr Coquimbo":                            "Corporación De Fomento De La Producción",
        # ── SII ───────────────────────────────────────────────────────────────
        "Servicio De Impuesto Interno":                 "Servicio De Impuestos Internos",
        "Servicios Impuestos Internos":                 "Servicio De Impuestos Internos",
        "Servicios De Impuestos Internos":              "Servicio De Impuestos Internos",
        # ── SERNAC ────────────────────────────────────────────────────────────
        "Servicio Nacionaldel Consumidor":              "Servicio Nacional Del Consumidor",
        "Sernac":                                       "Servicio Nacional Del Consumidor",
        # ── Tesorería General ─────────────────────────────────────────────────
        "Tesoreria General Provincial":                 "Tesorería General De La República",
        "Tesoreria General De La Republica":            "Tesorería General De La República",
        "Tesoreria General De La Republica - Oficina Provincial De Coquimbo": "Tesorería General De La República",
        "Tesoreria Regional Metropolitana":             "Tesorería General De La República",
        # ── Municipalidad La Serena ───────────────────────────────────────────
        "Ilustre Municipal De La Serena":               "Ilustre Municipalidad De La Serena",
        "I. Municipalidad De La Serena":                "Ilustre Municipalidad De La Serena",
        # ── Municipalidad Coquimbo ────────────────────────────────────────────
        "Municipalidad De Coquimbo":                    "Ilustre Municipalidad De Coquimbo",
        "Municipio De Coquimbo":                        "Ilustre Municipalidad De Coquimbo",
        # ── Otras municipalidades ─────────────────────────────────────────────
        "Ilustre Municipalidad De Vicuna":              "Ilustre Municipalidad De Vicuña",
        "Ilustre Municipalidad De Combarbala":          "Ilustre Municipalidad De Combarbalá",
        "I. Municipalidad De Monte Patria":             "Ilustre Municipalidad De Monte Patria",
        # ── Ripley ────────────────────────────────────────────────────────────
        "Ripley":                                       "Ripley Chile S.A.",
        "Ripley Store Spa":                             "Ripley Chile S.A.",
        "Ripley Store":                                 "Ripley Chile S.A.",
        "Ripley Store Ltda.":                           "Ripley Chile S.A.",
        "Ripley Coquimbo":                              "Ripley Chile S.A.",
        "Comercial Eccsa Ripley":                       "Ripley Chile S.A.",
        "Comercial Eccsa S.A":                          "Ripley Chile S.A.",
        "Comercial Eccsa S.A.":                         "Ripley Chile S.A.",
        # ── Enjoy ─────────────────────────────────────────────────────────────
        "Enjoy Casino Y Resort":                        "Enjoy S.A.",
        "Enjoy Coquimbo":                               "Enjoy S.A.",
        "Enjoy":                                        "Enjoy S.A.",
        "Campos Del Norte S.A":                         "Enjoy S.A.",
        "Campos Del Norte - Enjoy":                     "Enjoy S.A.",
        # ── Salfa / Salinas y Fabres ──────────────────────────────────────────
        "Salfa Salinas Y Fabres":                       "Salinas Y Fabres S.A.",
        "Salfa":                                        "Salinas Y Fabres S.A.",
        "Salinas Y Fabres (Salfa)":                     "Salinas Y Fabres S.A.",
        "Salinas Y Fabres S.A":                         "Salinas Y Fabres S.A.",
        "Salfa Salinas Y Fabres S.A.":                  "Salinas Y Fabres S.A.",
        # ── Techo ─────────────────────────────────────────────────────────────
        "Techo Para Chile":                             "Techo Chile",
        "Fundación Un Techo Para Chile":                "Techo Chile",
        # ── Otros duplicados por puntuación/tildes ────────────────────────────
        "Nestle Chile S.A":                             "Nestle Chile S.A.",
        "Aguas Del Valle S.A.":                         "Aguas Del Valle",
        "Afp Provida S,A,":                             "Afp Provida",
        "Embotelladora Andina S.A":                     "Embotelladora Andina S.A.",
        "Forum Servicios Financieros S.A":              "Forum Servicios Financieros S.A.",
        "Forum Servicios Financieros":                  "Forum Servicios Financieros S.A.",
        "Factoring Security":                           "Factoring Security S.A.",
        "Komatsu Cummins":                              "Komatsu Cummins Chile",
        "Telefonica Chile S.A.":                        "Telefonica Chile",
        "Terminal Puerto Coquimbo":                     "Terminal Puerto Coquimbo S.A.",
        "Transportes Ccu":                              "Transportes Ccu Limitada",
        "Rendic Hnos .S.A":                             "Rendic Hnos S.A.",
        "Paris Administradora Ltda.":                   "Paris Administradora Ltda",
        "Paris Administradora Limitada":                "Paris Administradora Ltda",
        "Fundación Chile":                              "Fundacion Chile",
        "Frigorifico Zepeda Y Corral Spa":              "Frigorífico Zepeda Y Corral Spa",
        "Ingenieria Y Maquinarias Real Spa":            "Ingeniería Y Maquinarias Real Spa",
        "Ingeniería Y Maquinarias Real":                "Ingeniería Y Maquinarias Real Spa",
        "Ingenieria Gestion Y Desarrollo S.A":          "Ingeniería, Gestión Y Desarrollo S.A",
        "Empresa Portuaria":                            "Empresa Portuaria Coquimbo",
        "Rodolfo Elias Morales Riones Automotora Spa":  "Rodolfo Elias Morales Briones Automotora Spa",
        "Rodolfo Elias Morales Briones  Automotora Spa": "Rodolfo Elias Morales Briones Automotora Spa",
        "Compania Industrial El Volcan S.A.":           "Compania Industrial El Volcan",
        "Compañia Industrial El Volcán S.A.":           "Compania Industrial El Volcan",
        "Compañia General De Electricidad":             "Compañia General De Electricidad S.A.",
        "Sociedad De Ingenieria Y Construccion Asseme Ltda": "Sociedad De Ingenieria Y Construccion Assieme Ltda",
        "Instituto Nacional De Estadisticas":           "Instituto Nacional De Estadistica",
        "Ip Proyectos  Industriales Spa":               "Ip Proyectos Industriales Spa",
        "Ip Proyectos Industriales Ltda":               "Ip Proyectos Industriales Spa",
        "Secretaria Regional Ministerial De Bienes Nacaionales": "Secretaria Regional Ministerial De Bienes Nacionales",
        "Secretaria Regional Ministerio De Bienes Nacionales":   "Secretaria Regional Ministerial De Bienes Nacionales",
        "Direccion Regional De Coquimbo (Aduana)":      "Direccion Regional De Aduana Coquimbo",
        "Direccion Regional De Aduana, Coquimbo":       "Direccion Regional De Aduana Coquimbo",
        "Direccion Regional Aduana":                    "Direccion Regional De Aduana Coquimbo",
        "Dirección Regional Aduana De Coquimbo":        "Direccion Regional De Aduana Coquimbo",
        "Centro De Desarrollo De Negocio":              "Centro De Desarrollo De Negocios",
        "Centro Desarrollo De Negocio":                 "Centro De Desarrollo De Negocios",
        "Centro Desarrollo De Negocios":                "Centro De Desarrollo De Negocios",
        "Centro De Desarrollo De Negocios La Serena":   "Centro De Desarrollo De Negocios",
        "Centro De Negocio Sercotec":                   "Centro De Negocios Sercotec",
        "Centro De Desarrollo De Negocios Sercotec":    "Centro De Negocios Sercotec",
        "Centro De Negocios Secotec":                   "Centro De Negocios Sercotec",
        "Servicio Local De Educacion Publica Puerto Cordillera":    "Servicio Local De Educación Pública Puerto Cordillera",
        "Servicio Local De Educacion Publica - Puerto Cordillera":  "Servicio Local De Educación Pública Puerto Cordillera",
        "Servicio Local De Educacion Publica Puerto Codillera":     "Servicio Local De Educación Pública Puerto Cordillera",
        "Asociacion De Exportadoes De Chile A.G. - Asoex": "Asociacion De Exportadores De Frutas De Chile A.G",
        "Asociación De Exportadores De Frutas De Chile  A.G": "Asociacion De Exportadores De Frutas De Chile A.G",
        "Callegari E Hijos Litda":                      "Callegari E Hijos Ltda",
        "Drilling Services And Solutions A.P.A":        "Drilling Services And Solutions Spa",
        "Agrosuper Comercializadora De Alimentos Ltda": "Agrosuper",
        "Agrosuper Com. Alim. Ltda":                    "Agrosuper",
    }

    # _normalizar_nombre es función de módulo (ver abajo de SIN_INFO)

    empresas = (
        df_pract[["nomb_empr_prac", "act_econo_empresa", "nat_empresa", "ciudad_emp_prac"]]
        .dropna(subset=["nomb_empr_prac"])
        .copy()
    )
    empresas["nombre"]     = empresas["nomb_empr_prac"].apply(_normalizar_nombre)
    empresas["rubro_norm"] = empresas["act_econo_empresa"].fillna("").astype(str).str.strip().str.title()

    n_antes = len(empresas)
    empresas = empresas.drop_duplicates(subset=["nombre"], keep="last").copy()
    n_despues = len(empresas)
    if n_antes - n_despues > 0:
        log.info("[TRANSFORM] dim_empresa: %d duplicados eliminados (%d → %d empresas únicas)",
                 n_antes - n_despues, n_antes, n_despues)

    def _sectores(nat):
        """
        Retorna 'Público', 'Privado' o 'Sin información' según la naturaleza jurídica.
        Nunca retorna NULL: el fallback es 'Sin información' para que Power BI
        pueda filtrar empresas no categorizadas con un slicer explícito.
        """
        v = str(nat).strip().lower().replace("á", "a").replace("ú", "u")
        if any(p in v for p in ["publica", "publico"]):
            return "Público"
        if str(nat).strip().lower() in ("nan", "none", "", "0"):
            return SIN_INFO
        return "Privado"

    # 'sectores': columna texto para filtros/slicers en Power BI (dim_empresa).
    # 'sector_publico' (1/0/NULL) se mantiene SOLO en fact_hechos para cálculos DAX.
    empresas["sectores"] = empresas["nat_empresa"].apply(_sectores)

    # disposicion_recibir se mantiene como atributo de empresa
    empresas["disposicion_recibir"] = 1
    empresas["id_rubro"] = empresas["rubro_norm"].map(mapa_rubro).fillna(id_sin_info).astype(int)
    empresas = empresas.reset_index(drop=True)
    empresas["id_empresa"] = empresas.index + 1

    # ── Localidad ─────────────────────────────────────────────────────────────
    # Se usa directamente ciudad_emp_prac de alumn_pract: campo "Ciudad" del
    # formulario de práctica, ingresado por el supervisor. Es la fuente más
    # confiable. Se normaliza a Title Case y se rellena con SIN_INFO si está vacío.
    def _normalizar_ciudad(ciudad) -> str:
        v = str(ciudad).strip()
        if v.lower() in ("nan", "none", ""):
            return SIN_INFO
        return v.title()

    # ciudad_emp_prac viene en la misma fila que el nombre de empresa;
    # al deduplicar por nombre (keep="last") queda la ciudad más reciente.
    empresas["localidad"] = empresas["ciudad_emp_prac"].apply(_normalizar_ciudad)

    n_sin_loc = (empresas["localidad"] == SIN_INFO).sum()
    if n_sin_loc > 0:
        log.warning(
            "[TRANSFORM] dim_empresa: %d empresa(s) sin ciudad registrada (ciudad_emp_prac vacío).",
            n_sin_loc,
        )

    # ── Coordenadas por ciudad ────────────────────────────────────────────────
    # Latitud y longitud exactas para cada ciudad de Chile que aparece en los datos.
    # Power BI las usa directamente en el mapa, sin ambigüedad geográfica.
    _COORDENADAS = {
        # Región de Coquimbo
        "La Serena":      (-29.9027, -71.2519),
        "Coquimbo":       (-29.9533, -71.3436),
        "Ovalle":         (-30.6010, -71.1997),
        "Illapel":        (-31.6348, -71.1686),
        "Salamanca":      (-31.7758, -70.9726),
        "Los Vilos":      (-31.9094, -71.5090),
        "Andacollo":      (-30.2286, -71.0847),
        "Vicuña":         (-30.0321, -70.7104),
        "Monte Patria":   (-30.6957, -70.9643),
        "Combarbalá":     (-31.1822, -71.0278),
        "Punitaqui":      (-30.8333, -71.2667),
        "Canela":         (-31.3978, -71.4508),
        "Río Hurtado":    (-30.4500, -70.8167),
        "La Higuera":     (-29.5000, -71.2500),
        "Paiguano":       (-30.0500, -70.5000),
        # Región de Atacama
        "Copiapó":        (-27.3667, -70.3333),
        "Vallenar":       (-28.5731, -70.7597),
        "Chañaral":       (-26.3456, -70.6217),
        "Caldera":        (-27.0653, -70.7956),
        "Huasco":         (-28.4667, -71.2167),
        "Freirina":       (-28.5000, -71.0833),
        "Diego De Almagro": (-26.3667, -70.0500),
        "Tierra Amarilla":(-27.4833, -70.2833),
        # Región de Valparaíso
        "Valparaíso":     (-33.0472, -71.6127),
        "Viña Del Mar":   (-33.0245, -71.5518),
        "San Antonio":    (-33.5928, -71.6072),
        "Los Andes":      (-32.8337, -70.5997),
        "Quillota":       (-32.8797, -71.2467),
        "San Felipe":     (-32.7500, -70.7167),
        "La Ligua":       (-32.4500, -71.2333),
        "Casablanca":     (-33.3167, -71.4167),
        "Quilpué":        (-33.0500, -71.4333),
        "Villa Alemana":  (-33.0431, -71.3736),
        "Concón":         (-32.9167, -71.5333),
        "La Calera":      (-32.7833, -71.2000),
        # Región Metropolitana
        "Santiago":       (-33.4569, -70.6483),
        "Providencia":    (-33.4333, -70.6167),
        "Las Condes":     (-33.4167, -70.5833),
        "Maipú":          (-33.5117, -70.7581),
        "Puente Alto":    (-33.6117, -70.5758),
        "San Bernardo":   (-33.5928, -70.6989),
        "Vitacura":       (-33.3939, -70.5781),
        "Lo Barnechea":   (-33.3528, -70.5167),
        "Ñuñoa":          (-33.4528, -70.5972),
        "La Florida":     (-33.5167, -70.5833),
        "Peñalolén":      (-33.4833, -70.5333),
        "Quilicura":      (-33.3667, -70.7333),
        "Melipilla":      (-33.6928, -71.2117),
        "Talagante":      (-33.6667, -70.9333),
        "Colina":         (-33.2000, -70.6833),
        "Padre Hurtado":  (-33.5667, -70.8167),
        # Región de O'Higgins
        "Rancagua":       (-34.1703, -70.7403),
        "San Fernando":   (-34.5833, -71.0000),
        "Santa Cruz":     (-34.6333, -71.3667),
        # Región del Maule
        "Talca":          (-35.4264, -71.6553),
        "Curicó":         (-34.9833, -71.2333),
        "Linares":        (-35.8500, -71.5833),
        "Constitución":   (-35.3333, -72.4167),
        # Región de Ñuble
        "Chillán":        (-36.6067, -72.1033),
        "Chillán Viejo":  (-36.6333, -72.1167),
        # Región del Biobío
        "Concepción":     (-36.8201, -73.0444),
        "Talcahuano":     (-36.7167, -73.1167),
        "Los Ángeles":    (-37.4694, -72.3528),
        "Coronel":        (-37.0167, -73.1500),
        "Tomé":           (-36.6167, -72.9500),
        "Lota":           (-37.0833, -73.1667),
        "Cañete":         (-37.8000, -73.4000),
        "Lebu":           (-37.6167, -73.6500),
        # Región de La Araucanía
        "Temuco":         (-38.7359, -72.5904),
        "Villarrica":     (-39.2833, -72.2333),
        "Pucón":          (-39.2833, -71.9667),
        "Angol":          (-37.7964, -72.7083),
        "Nueva Imperial": (-38.7431, -72.9583),
        # Región de Los Ríos
        "Valdivia":       (-39.8142, -73.2459),
        "La Unión":       (-40.2833, -73.0833),
        "Panguipulli":    (-39.6417, -72.3333),
        "Río Bueno":      (-40.3333, -72.9667),
        # Región de Los Lagos
        "Puerto Montt":   (-41.4717, -72.9367),
        "Osorno":         (-40.5731, -73.1347),
        "Puerto Varas":   (-41.3167, -72.9833),
        "Castro":         (-42.4797, -73.7608),
        "Ancud":          (-41.8667, -73.8333),
        "Frutillar":      (-41.1333, -73.0500),
        # Región de Aysén
        "Coyhaique":      (-45.5714, -72.0664),
        "Puerto Natales": (-51.7333, -72.5000),
        # Región de Magallanes
        "Punta Arenas":   (-53.1638, -70.9171),
        # Región de Arica y Parinacota
        "Arica":          (-18.4783, -70.3225),
        # Región de Tarapacá
        "Iquique":        (-20.2208, -70.1431),
        # Región de Antofagasta
        "Antofagasta":    (-23.6500, -70.4000),
        "Calama":         (-22.4667, -68.9333),
        "Tocopilla":      (-22.0917, -70.1972),
        "Sierra Gorda":   (-22.8972, -69.3208),
    }

    def _asignar_coords(localidad: str):
        return _COORDENADAS.get(localidad, (None, None))

    empresas[["latitud", "longitud"]] = pd.DataFrame(
        empresas["localidad"].apply(_asignar_coords).tolist(),
        index=empresas.index,
    )

    df = empresas[["id_empresa", "nombre", "localidad", "latitud", "longitud",
                   "sectores", "disposicion_recibir", "id_rubro"]].copy()
    _guardar(df, "dim_empresa")
    return df


def construir_dim_carrera(df_pract: pd.DataFrame) -> pd.DataFrame:
    carreras = ["Ingeniería Comercial", "Ingeniería en Información y Control de Gestión"]
    df = pd.DataFrame({"id_carrera": range(1, len(carreras) + 1), "nombre_carrera": carreras})
    _guardar(df, "dim_carrera")
    return df


def construir_dim_estudiante(df_pract: pd.DataFrame, df_carrera: pd.DataFrame) -> pd.DataFrame:
    mapa_carrera = dict(zip(df_carrera["nombre_carrera"], df_carrera["id_carrera"]))

    est = df_pract.copy()
    est["fech_ini_prac"] = est["fech_ini_prac"].apply(_parse_fecha)
    est = est.sort_values(by="fech_ini_prac", ascending=True)

    est["rut_alumno"]   = est["rut_alum"].astype(str).apply(_normalizar_rut)
    est["nombre"]       = est["nomb_alum"].astype(str).str.strip().str.title()
    est["apellido"]     = est["apell_alum"].astype(str).str.strip().str.title()
    est["carrera_norm"] = est["carr_alum"].astype(str).apply(_normalizar_carrera)
    est["id_carrera"]   = est["carrera_norm"].map(mapa_carrera).fillna(1).astype(int)

    for col_orig, col_dest in [("fono_alum", "telefono"), ("email_alum", "correo")]:
        if col_orig in est.columns:
            est[col_dest] = est[col_orig].astype(str).str.strip().replace({"nan": None, "None": None})
        else:
            est[col_dest] = None

    # Descartar estudiantes sin RUT válido antes de deduplicar
    est = est[est["rut_alumno"] != ""].copy()

    # Descartar estudiantes cuya carrera no pudo ser identificada.
    # Estos registros generarían blancos en cualquier segmentación por carrera.
    n_sin_carrera = (est["carrera_norm"] == SIN_INFO).sum()
    if n_sin_carrera > 0:
        log.warning(
            "[TRANSFORM] dim_estudiante: %d estudiante(s) descartado(s) por "
            "carrera no identificable (valor='%s')", n_sin_carrera, SIN_INFO
        )
    est = est[est["carrera_norm"] != SIN_INFO].copy()

    est = est.drop_duplicates(subset=["rut_alumno"], keep="last").reset_index(drop=True)
    est["id_estudiante"] = est.index + 1

    df = est[["id_estudiante", "rut_alumno", "nombre", "apellido",
              "telefono", "correo", "id_carrera"]].copy()
    _guardar(df, "dim_estudiante")
    return df


def construir_dim_tiempo(df_pract: pd.DataFrame) -> pd.DataFrame:
    fechas = df_pract["fech_ini_prac"].apply(_parse_fecha).dropna().unique()
    fechas = sorted(set(pd.Timestamp(f).date() for f in fechas))

    registros = []
    for fecha in fechas:
        ts = pd.Timestamp(fecha)
        registros.append({
            "fecha":      fecha,
            "mes":        ts.month,
            "nombre_mes": ts.strftime("%B").capitalize(),
            "semestre":   1 if ts.month <= 6 else 2,
            "año":        ts.year,
        })

    df = pd.DataFrame(registros).reset_index(drop=True)
    df.insert(0, "id_tiempo", df.index + 1)
    _guardar(df, "dim_tiempo")
    return df


def construir_dim_evaluacion(df_eva_jef: pd.DataFrame) -> pd.DataFrame:
    """
    Construye dim_evaluacion_empresa SIN comentarios.
    Los comentarios se procesan en vcm_comentarios.py y se integran en la carga.
    Incluye las respuestas individuales r_2_1..r_2_10 como columnas INTEGER.
    """
    # A=4, B=3, C=2, D=1  (coincide con la escala 4→1 del formulario web)
    mapa_letra     = {"A": 4, "B": 3, "C": 2, "D": 1}
    cols_presentes = [c for c in COLS_ITEMS if c in df_eva_jef.columns]

    # Convertir respuestas letra→número entero para cada columna individual
    df_resp = df_eva_jef[cols_presentes].replace(mapa_letra)
    df_resp = df_resp.apply(pd.to_numeric, errors="coerce")

    # Calificación promedio (escala 1–4)
    calificaciones = df_resp.mean(axis=1).round(2)

    df = pd.DataFrame({
        "rut_alum":          df_eva_jef["rut_alum"].astype(str).str.strip().apply(_normalizar_rut).values,
        "calificacion":      calificaciones.values,
        "comentario":        None,
        "sentimiento":       None,
        "score_sentimiento": None,
        "categoria_brecha":  None,
    })

    for col in COLS_ITEMS:
        if col in cols_presentes:
            df[col] = df_resp[col].where(df_resp[col].notna(), None)
            df[col] = pd.array(df[col], dtype=pd.Int64Dtype())
        else:
            df[col] = None

    df = df.reset_index(drop=True)
    df.insert(0, "id_evaluacion", df.index + 1)

    # Descartar evaluaciones donde la calificación es NaN porque TODOS los
    # ítems estaban vacíos. Una evaluación sin ninguna respuesta no aporta
    # información y generaría blancos en los gráficos de calificación.
    n_total = len(df)
    df = df.dropna(subset=["calificacion"]).reset_index(drop=True)
    df["id_evaluacion"] = df.index + 1  # reasignar IDs consecutivos tras el filtro
    n_descartadas = n_total - len(df)
    if n_descartadas > 0:
        log.warning(
            "[TRANSFORM] dim_evaluacion_empresa: %d evaluación(es) descartada(s) "
            "porque todos sus ítems estaban vacíos (calificacion=NULL)",
            n_descartadas,
        )

    _guardar(df, "dim_evaluacion_empresa")
    return df


def construir_dim_practica(df_pract: pd.DataFrame, df_eva: pd.DataFrame) -> pd.DataFrame:
    """
    dim_practica tiene id_evaluacion como FK y estado_practica como atributo.

    Lógica de estado:
      - Se ordenan las prácticas por fecha de inicio para cada estudiante.
      - Las prácticas anteriores a la última de un estudiante se marcan como
        "reprobada" (fue necesario repetirla), independientemente de lo que
        diga el campo proces_terminado.
      - Para la práctica más reciente (o única) de cada estudiante se usa el
        valor real del campo proces_terminado: "aprobada", "reprobada" o
        "en_proceso".
    """
    def _estado_raw(val: str) -> str:
        """Convierte el valor crudo de proces_terminado a un estado canónico."""
        v = str(val).strip().lower()
        if v in ("1", "true", "sí", "si", "aprobado", "aprobada"):
            return "aprobada"
        if v in ("0", "false", "no", "reprobado", "reprobada"):
            return "reprobada"
        return "en_proceso"

    # Identificar la práctica más reciente por estudiante (por fecha de inicio)
    pract_work = df_pract.copy()
    pract_work["_fech_ini"] = pract_work["fech_ini_prac"].apply(_parse_fecha)
    pract_work["_rut_norm"] = pract_work["rut_alum"].astype(str).apply(_normalizar_rut)
    pract_work = pract_work.reset_index(drop=True)
    pract_work["_idx_orig"] = pract_work.index

    # Índice de la práctica más reciente por RUT
    idx_ultima = (
        pract_work.sort_values("_fech_ini", ascending=True)
                  .groupby("_rut_norm")["_idx_orig"]
                  .last()
    )
    set_ultimas = set(idx_ultima.values)

    def _estado(row) -> str:
        idx_orig = row.name  # índice original tras reset_index
        es_ultima = idx_orig in set_ultimas
        if es_ultima:
            # Práctica más reciente: usar el estado real
            return _estado_raw(row.get("proces_terminado", ""))
        else:
            # Práctica anterior: el estudiante tuvo que repetirla → reprobada
            return "reprobada"

    mapa_eval      = {}
    mapa_eval_body = {}
    if "rut_alum" in df_eva.columns:
        for _, row in df_eva.iterrows():
            rut_jef = _normalizar_rut(str(row["rut_alum"]))
            id_e    = int(row["id_evaluacion"])
            if rut_jef and rut_jef not in mapa_eval:
                mapa_eval[rut_jef]      = id_e
                mapa_eval_body[rut_jef] = id_e

    pract = pract_work.copy()
    pract["estado_practica"] = pract.apply(_estado, axis=1)
    pract = pract.reset_index(drop=True)
    pract["id_practica"] = pract.index + 1
    pract["rut_norm"]    = pract["rut_alum"].astype(str).apply(_normalizar_rut)
    pract["rut_body"]    = pract["rut_alum"].astype(str).apply(_rut_cuerpo)
    pract["id_evaluacion"] = pract["rut_norm"].map(mapa_eval).fillna(
                             pract["rut_body"].map(mapa_eval_body))

    df = pract[["id_practica", "id_evaluacion", "estado_practica"]].copy()
    _guardar(df, "dim_practica")
    return df


# ─── Tabla de hechos ──────────────────────────────────────────────────────────

def construir_fact_hechos(df_pract, df_estudiante, df_practica,
                           df_empresa, df_tiempo, df_evaluacion, df_rubro):
    log.info("[TRANSFORM] Construyendo tabla de hechos...")

    mapa_est        = dict(zip(df_estudiante["rut_alumno"],   df_estudiante["id_estudiante"]))
    mapa_emp        = dict(zip(df_empresa["nombre"],           df_empresa["id_empresa"]))
    mapa_tpo        = dict(zip(df_tiempo["fecha"].astype(str), df_tiempo["id_tiempo"]))
    mapa_rubro      = dict(zip(df_empresa["id_empresa"],       df_empresa["id_rubro"]))
    # sector_publico en fact_hechos se deriva del texto 'sectores' de dim_empresa:
    # 'Público' → 1, 'Privado' → 0, 'Sin información' → NULL
    def _sector_a_int(s):
        if s == "Público":  return 1
        if s == "Privado":  return 0
        return None
    mapa_sector_int = {id_e: _sector_a_int(sec)
                       for id_e, sec in zip(df_empresa["id_empresa"], df_empresa["sectores"])}
    mapa_rubro_desc = dict(zip(df_rubro["id_rubro"],           df_rubro["descripcion_rubro"]))

    # Mapa rut → id_evaluacion con fallback a cuerpo de RUT
    mapa_eval_rut      = {}
    mapa_eval_rut_body = {}
    if "rut_alum" in df_evaluacion.columns:
        for _, row in df_evaluacion.iterrows():
            rut_jef = _normalizar_rut(str(row["rut_alum"]))
            id_e    = int(row["id_evaluacion"])
            if rut_jef and rut_jef not in mapa_eval_rut:
                mapa_eval_rut[rut_jef]      = id_e
                mapa_eval_rut_body[rut_jef] = id_e

    # [FIX 3] Detectar estudiantes con más de una práctica registrada
    conteo_rut     = df_pract["rut_alum"].astype(str).apply(_normalizar_rut).value_counts()
    ruts_repetidos = set(conteo_rut[conteo_rut > 1].index)

    RUBROS_PERTINENTES = {
        "administración y servicios", "finanzas y contabilidad", "educación",
        "salud", "tecnología e informática", "agricultura y pesca",
        "industria y manufactura", "comercio",
    }

    # Precargar ítems individuales desde df_evaluacion
    mapa_items = {}
    for _, row in df_evaluacion.iterrows():
        id_e = int(row["id_evaluacion"])
        mapa_items[id_e] = {col: (None if pd.isna(row[col]) else int(row[col]))
                            for col in COLS_ITEMS if col in df_evaluacion.columns}

    hechos = []
    descartados_claves = 0  # [REQ 12] contador de filas descartadas por claves nulas

    for idx, row in df_pract.iterrows():
        rut        = _normalizar_rut(str(row.get("rut_alum", "")))
        nombre_emp = _normalizar_nombre(str(row.get("nomb_empr_prac", "")).strip())
        fecha_raw  = _parse_fecha(row.get("fech_ini_prac"))
        fecha_str  = str(fecha_raw.date()) if fecha_raw else None

        id_estudiante = mapa_est.get(rut)
        id_practica   = idx + 1
        id_empresa    = mapa_emp.get(nombre_emp)
        id_tiempo     = mapa_tpo.get(fecha_str)
        id_evaluacion = mapa_eval_rut.get(rut) or mapa_eval_rut_body.get(_rut_cuerpo(rut))

        # [REQ 12] Descartar filas con claves dimensionales obligatorias nulas.
        # Una fila sin id_estudiante o sin id_empresa no tiene sentido en el
        # modelo dimensional y generaría blancos no filtrables en Power BI.
        if not id_estudiante or not id_empresa:
            descartados_claves += 1
            log.debug("[TRANSFORM] Fila %d descartada: id_estudiante=%s, id_empresa=%s",
                      idx, id_estudiante, id_empresa)
            continue

        # id_tiempo puede ser nulo (fecha de práctica ausente): se registra como NULL
        # en lugar de descartar la fila, ya que el resto de métricas sigue siendo válido.

        # ── Métricas ─────────────────────────────────────────────────────────

        # Pertinencia: rubro de la empresa pertenece a áreas afines a las carreras
        id_rubro_emp = mapa_rubro.get(id_empresa)
        desc_rubro   = mapa_rubro_desc.get(id_rubro_emp, "").lower()
        pertinencia  = int(any(p in desc_rubro for p in RUBROS_PERTINENTES))

        # [REQ 8] sector_publico: INTEGER desde dim_empresa (1, 0 o NULL)
        sector_publico = mapa_sector_int.get(id_empresa)

        # Aprobación
        estado     = str(row.get("proces_terminado", "")).strip().lower()
        aprobacion = 1 if estado in ("1", "true", "sí", "si", "aprobado", "aprobada") else 0

        # [FIX 1] Bidireccionalidad real: ¿la empresa devolvió evaluación?
        bidireccionalidad = 1 if id_evaluacion is not None else 0

        # [FIX 3] ¿Este estudiante hizo la práctica más de una vez?
        practica_repetida = 1 if rut in ruts_repetidos else 0

        # Calificación empresa
        eval_row             = df_evaluacion[df_evaluacion["id_evaluacion"] == id_evaluacion]
        calificacion_empresa = float(eval_row["calificacion"].values[0]) if not eval_row.empty else None

        # Índices sentimiento (NULL hasta que vcm_comentarios.py los rellene)
        indice_positivo  = None
        indice_negativo  = None
        indice_neutro    = None
        categoria_brecha = None

        # [FIX 2] Ítems individuales de la evaluación propagados a fact_hechos
        items_eval = mapa_items.get(id_evaluacion, {col: None for col in COLS_ITEMS})

        # Nivel de estudio (semestre en que realiza la práctica)
        nivel_estudio = _normalizar_nivel_estudio(row.get("niv_estudio_alum"))

        # Periodicidad: verano (dic-mar) o invierno (abr-nov)
        periodicidad = _calcular_periodicidad(fecha_raw)

        registro = {
            "id_hechos":            len(hechos) + 1,
            "id_practica":          id_practica,
            "id_estudiante":        id_estudiante,
            "id_tiempo":            id_tiempo,
            "id_empresa":           id_empresa,
            "pertinencia":          pertinencia,
            "sector_publico":       sector_publico,
            "aprobacion":           aprobacion,
            "bidireccionalidad":    bidireccionalidad,
            "practica_repetida":    practica_repetida,
            "calificacion_empresa": calificacion_empresa,
            "nivel_estudio":        nivel_estudio,
            "periodicidad":         periodicidad,
            "indice_positivo":      indice_positivo,
            "indice_negativo":      indice_negativo,
            "indice_neutro":        indice_neutro,
            "categoria_brecha":     categoria_brecha,
        }
        for col in COLS_ITEMS:
            registro[col] = items_eval.get(col)

        # Incluir TODAS las prácticas en fact_hechos:
        # - Con evaluación → bidireccionalidad=1
        # - Sin evaluación → bidireccionalidad=0 (empresa no respondió)
        hechos.append(registro)

    if descartados_claves > 0:
        log.warning("[TRANSFORM] %d filas descartadas de fact_hechos por "
                    "id_estudiante o id_empresa nulos (datos insuficientes en la fuente)",
                    descartados_claves)

    df = pd.DataFrame(hechos)
    log.info("[TRANSFORM] Tabla de hechos: %d registros (de %d práctica(s) fuente)",
             len(df), len(df_pract))

    # ── Reporte de campos con blancos tolerados (pendientes de scraping) ──────
    # sector_publico: empresas cuya naturaleza jurídica no está en la API.
    #   → Se completará con scraping (Registro de Comercio / SII).
    #   → Estos registros SÍ se cargan; el campo queda NULL hasta el scraping.
    n_sin_sector = df["sector_publico"].isna().sum()
    if n_sin_sector > 0:
        log.warning(
            "[TRANSFORM] fact_hechos: %d fila(s) con sector_publico=NULL "
            "(pendiente scraping de naturaleza jurídica de empresa)",
            n_sin_sector,
        )

    # id_evaluacion / bidireccionalidad=0: práctica sin evaluación de la empresa.
    #   → Se completará si se obtiene la evaluación por scraping o fuente externa.
    #   → Estos registros SÍ se cargan con bidireccionalidad=0.
    n_sin_eval = (df["bidireccionalidad"] == 0).sum()
    if n_sin_eval > 0:
        log.warning(
            "[TRANSFORM] fact_hechos: %d fila(s) con bidireccionalidad=0 "
            "(empresa no entregó evaluación — pendiente scraping o fuente externa)",
            n_sin_eval,
        )

    _guardar(df, "fact_hechos")
    return df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    log.info("[TRANSFORM] Leyendo archivos crudos...")
    df_pract   = _leer_raw("alumn_pract")
    df_eva_jef = _leer_raw("alumn_pract_eva_jef")

    log.info("[TRANSFORM] Construyendo dimensiones...")
    df_rubro      = construir_dim_rubro(df_pract)
    df_empresa    = construir_dim_empresa(df_pract, df_rubro)
    df_carrera    = construir_dim_carrera(df_pract)
    df_estudiante = construir_dim_estudiante(df_pract, df_carrera)
    df_tiempo     = construir_dim_tiempo(df_pract)
    df_evaluacion = construir_dim_evaluacion(df_eva_jef)
    df_practica   = construir_dim_practica(df_pract, df_evaluacion)

    construir_fact_hechos(
        df_pract, df_estudiante, df_practica,
        df_empresa, df_tiempo, df_evaluacion, df_rubro,
    )

    log.info("[TRANSFORM] Transformación completada ✔ — archivos en: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()