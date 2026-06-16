import os
import logging
import pandas as pd
from pysentimiento import create_analyzer
import spacy

DATA_DIR   = "data"
OUTPUT_DIR = os.path.join(DATA_DIR, "transformado")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

CATEGORIAS_BRECHA = {
    "herramientas_tecnologicas": [
        "excel", "software", "sistema", "computador", "tecnologia",
        "programa", "office", "digital", "plataforma", "herramienta",
        "aplicacion", "base de datos", "erp", "sap", "autocad",
    ],
    "competencias_blandas": [
        "comunicacion", "trabajo en equipo", "puntualidad", "actitud",
        "responsabilidad", "iniciativa", "liderazgo", "adaptacion", "proactividad",
        "empatia", "respeto", "colaboracion", "disposicion", "compromiso",
    ],
    "conocimientos_tecnicos": [
        "conocimiento", "teoria", "practica", "fundamento", "base",
        "formacion", "preparacion", "tecnico", "profesional", "disciplina",
        "contabilidad", "finanzas", "auditoria", "tributario", "costos",
        "marketing", "ventas", "administracion", "gestion",
    ],
    "habilidades_redaccion": [
        "redaccion", "informe", "escritura", "ortografia", "documento",
        "reporte", "comunicacion escrita", "expresion escrita", "sintaxis",
    ],
    "puntualidad_asistencia": [
        "tardanza", "ausencia", "horario", "puntual", "llegar tarde",
        "faltar", "inasistencia", "cumplimiento horario", "atraso",
    ],
    "iniciativa_proactividad": [
        "espera instrucciones", "autonomia", "propio", "sin pedir",
        "tomar decision", "resolver solo", "proponer", "sugerir",
    ],
    "atencion_cliente": [
        "cliente", "publico", "usuario", "atencion", "trato",
        "servicio al cliente", "amabilidad", "cordialidad",
    ],
    "resolucion_problemas": [
        "resolver", "problema", "solucion", "decision", "criterio",
        "analisis", "diagnostico", "improvisto",
    ],
}


# ─── Etapas NLP explícitas ────────────────────────────────────────────────────

def _tokenizar(texto: str, nlp) -> list:
    """
    Etapa 1 — Tokenización.
    Divide el texto en tokens individuales usando el modelo spaCy.
    Retorna la lista de objetos Token con toda su información lingüística.
    """
    return list(nlp(str(texto).lower()[:1000]))


def _filtrar_stop_words(tokens: list) -> list:
    """
    Etapa 2 — Eliminación de stop words.
    Descarta tokens que sean palabras vacías (artículos, preposiciones, etc.)
    o signos de puntuación, que no aportan información semántica.
    """
    return [t for t in tokens if not t.is_stop and not t.is_punct and t.text.strip()]


def _lematizar(tokens: list) -> str:
    """
    Etapa 3 — Lematización.
    Reduce cada token a su forma canónica (lema). Ej: "trabajando" → "trabajar".
    Retorna un string con todos los lemas separados por espacio.
    """
    return " ".join(t.lemma_ for t in tokens)


def _chunking(doc) -> str:
    """
    Etapa 4 — Chunking (extracción de sintagmas nominales).
    Identifica grupos nominales compuestos (noun_chunks) en el documento.
    Esto captura términos multi-palabra como "trabajo en equipo" o
    "comunicación escrita" que la lematización individual no detecta.
    Retorna un string con todos los chunks en minúscula.
    """
    return " ".join(chunk.text.lower() for chunk in doc.noun_chunks)


# ─── NLP ──────────────────────────────────────────────────────────────────────

def analizar_sentimiento(texto: str, analizador) -> tuple:
    """
    Analiza el sentimiento de un texto usando pysentimiento.
    Asume que el texto ya fue validado (no vacío) por el llamador.
    Retorna (sentimiento: str, score: float).
    """
    try:
        resultado   = analizador.predict(str(texto)[:512])
        etiqueta    = resultado.output.lower()
        mapa        = {"pos": "positivo", "neg": "negativo", "neu": "neutro"}
        sentimiento = mapa.get(etiqueta, "neutro")
        prob        = resultado.probas.get(resultado.output, 0.5)
        score       = prob if sentimiento == "positivo" else (-prob if sentimiento == "negativo" else 0.0)
        return sentimiento, round(score, 4)
    except Exception as e:
        log.warning("[NLP] Error sentimiento: %s", e)
        return "neutro", 0.0


def _quitar_tildes(texto: str) -> str:
    """Normaliza texto eliminando tildes para mejorar coincidencias."""
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "a", "É": "e", "Í": "i", "Ó": "o", "Ú": "u",
        "ü": "u", "Ü": "u", "ñ": "n", "Ñ": "n",
    }
    for orig, rep in reemplazos.items():
        texto = texto.replace(orig, rep)
    return texto


def detectar_brecha(texto: str, nlp) -> str:
    """
    Clasifica el comentario en una categoría de brecha usando el pipeline
    NLP completo: tokenización → stop words → lematización → chunking.

    La búsqueda opera sobre:
      - lemas individuales: para capturar variantes morfológicas.
      - chunks nominales  : para capturar términos compuestos.
    """
    try:
        # ── Etapa 1: Tokenización ─────────────────────────────────────────────
        doc    = nlp(str(texto).lower()[:1000])
        tokens = list(doc)
        log.debug("[NLP] Tokens: %d", len(tokens))

        # ── Etapa 2: Eliminación de stop words ───────────────────────────────
        tokens_limpios = _filtrar_stop_words(tokens)
        log.debug("[NLP] Tokens sin stop words: %d", len(tokens_limpios))

        # ── Etapa 3: Lematización ─────────────────────────────────────────────
        texto_lematizado = _lematizar(tokens_limpios)
        log.debug("[NLP] Texto lematizado: %s", texto_lematizado)

        # ── Etapa 4: Chunking ─────────────────────────────────────────────────
        texto_chunks = _chunking(doc)
        log.debug("[NLP] Chunks nominales: %s", texto_chunks)

        # Superficie de búsqueda = lemas + chunks normalizados sin tildes
        superficie = _quitar_tildes(texto_lematizado + " " + texto_chunks)

        for categoria, palabras in CATEGORIAS_BRECHA.items():
            if any(p in superficie for p in palabras):
                return categoria
        return "sin_brecha"

    except Exception as e:
        log.warning("[NLP] Error brecha: %s", e)
        return "sin_brecha"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _indices_sentimiento(sentimiento: str) -> tuple:
    """Retorna (indice_positivo, indice_negativo, indice_neutro) como enteros 0/1."""
    return (
        int(sentimiento == "positivo"),
        int(sentimiento == "negativo"),
        int(sentimiento == "neutro"),
    )


def _texto_valido(texto) -> bool:
    """
    [REQ 1] Determina si un comentario tiene contenido real para procesar.
    Retorna False para nulos, cadenas vacías o placeholders sin información.
    """
    if texto is None:
        return False
    s = str(texto).strip()
    return s not in ("", "nan", "None", "N/A", "n/a", "-", ".")


def _ruta(nombre: str) -> str:
    return os.path.join(OUTPUT_DIR, f"{nombre}.csv")


def _verificar_archivo(ruta: str, paso_previo: str):
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No encontrado: {ruta}. Ejecuta {paso_previo} primero."
        )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Leer comentarios crudos ────────────────────────────────────────────
    ruta_jef = os.path.join(DATA_DIR, "alumn_pract_eva_jef.csv")
    _verificar_archivo(ruta_jef, "vcm_extraccion.py")

    df_jef = pd.read_csv(ruta_jef, low_memory=False)
    comentarios_raw = (
        df_jef["r_2_13"].fillna("").astype(str).str.strip()
        if "r_2_13" in df_jef.columns
        else pd.Series([""] * len(df_jef))
    )
    total_raw = len(comentarios_raw)

    # [REQ 1] Descarte obligatorio de registros sin comentario real.
    # Se identifican los índices con contenido válido para el NLP.
    # Los índices SIN comentario reciben sentimiento=None y brecha=None
    # en la dimensión, sin pasar por los modelos.
    mask_validos = comentarios_raw.apply(_texto_valido)
    n_validos    = mask_validos.sum()
    n_descartados = total_raw - n_validos

    log.info("[COMENTARIOS] Total registros        : %d", total_raw)
    log.info("[COMENTARIOS] Con comentario válido  : %d (serán procesados)", n_validos)
    log.info("[COMENTARIOS] Sin comentario (blancos): %d (descartados del NLP)", n_descartados)

    # ── 2. Cargar modelos NLP ─────────────────────────────────────────────────
    log.info("[NLP] Cargando modelos...")
    analizador = create_analyzer(task="sentiment", lang="es")
    nlp        = spacy.load("es_core_news_sm")
    log.info("[NLP] Modelos cargados ✔")

    # ── 3. Procesar SOLO los comentarios válidos ──────────────────────────────
    # Se inicializan vectores de resultado con valores nulos para todos
    # los índices; solo los válidos son sobreescritos por el análisis NLP.
    sentimientos = [None] * total_raw
    scores       = [None] * total_raw
    categorias   = [None] * total_raw

    indices_validos = comentarios_raw[mask_validos].index.tolist()
    total_a_procesar = len(indices_validos)

    for i, idx in enumerate(indices_validos):
        if i % 50 == 0:
            log.info("  [NLP] %d / %d procesados...", i, total_a_procesar)

        texto = comentarios_raw[idx]

        # Pipeline NLP: los 4 pasos están implementados dentro de
        # analizar_sentimiento y detectar_brecha. El flujo es:
        #   1. Tokenización → 2. Stop words → 3. Lematización → 4. Chunking
        s, sc = analizar_sentimiento(texto, analizador)
        c     = detectar_brecha(texto, nlp)

        sentimientos[idx] = s
        scores[idx]       = sc
        categorias[idx]   = c

    log.info("[NLP] Procesamiento completado: %d comentarios analizados", total_a_procesar)

    # ── 4. Actualizar dim_evaluacion_empresa con resultados NLP ───────────────
    ruta_eval = _ruta("dim_evaluacion_empresa")
    _verificar_archivo(ruta_eval, "vcm_transformacion.py")

    df_eval = pd.read_csv(ruta_eval)

    # Solo se asignan comentarios a los registros con texto real;
    # los blancos quedan con NULL en comentario, sentimiento y brecha.
    comentarios_limpios = comentarios_raw.where(mask_validos, other=None)

    df_eval["comentario"]        = comentarios_limpios.values[:len(df_eval)]
    df_eval["sentimiento"]       = sentimientos[:len(df_eval)]
    df_eval["score_sentimiento"] = scores[:len(df_eval)]
    df_eval["categoria_brecha"]  = categorias[:len(df_eval)]

    df_eval.to_csv(ruta_eval, index=False, encoding="utf-8")
    log.info("[NLP] dim_evaluacion_empresa actualizada ✔")

    # ── 5. Actualizar fact_hechos con índices de sentimiento y brecha ─────────
    #
    # El join correcto es:
    #   fact_hechos.id_practica → dim_practica.id_practica
    #   → dim_practica.id_evaluacion → dim_evaluacion_empresa.id_evaluacion

    ruta_hechos   = _ruta("fact_hechos")
    ruta_practica = _ruta("dim_practica")
    _verificar_archivo(ruta_hechos,   "vcm_transformacion.py")
    _verificar_archivo(ruta_practica, "vcm_transformacion.py")

    df_hechos       = pd.read_csv(ruta_hechos)
    df_pract_bridge = pd.read_csv(ruta_practica)[["id_practica", "id_evaluacion"]]

    # Merge para obtener id_evaluacion en cada fila de fact_hechos
    df_hechos = df_hechos.merge(
        df_pract_bridge,
        on="id_practica",
        how="left",
        suffixes=("", "_bridge"),
    )

    if "id_evaluacion_bridge" in df_hechos.columns:
        df_hechos["id_evaluacion"] = df_hechos["id_evaluacion_bridge"]
        df_hechos = df_hechos.drop(columns=["id_evaluacion_bridge"])

    # Construir mapas desde df_eval (ya con NLP procesado)
    # Solo los id_evaluacion con sentimiento real (no None) participan.
    df_eval_con_sent = df_eval.dropna(subset=["sentimiento"])
    mapa_sentimiento = dict(zip(df_eval_con_sent["id_evaluacion"], df_eval_con_sent["sentimiento"]))
    mapa_brecha      = dict(zip(df_eval_con_sent["id_evaluacion"], df_eval_con_sent["categoria_brecha"]))

    def _get_indices(id_eval):
        if pd.isna(id_eval) or id_eval not in mapa_sentimiento:
            # Sin evaluación o comentario descartado: índices a NULL
            return pd.Series([None, None, None])
        sent = mapa_sentimiento[id_eval]
        return pd.Series(_indices_sentimiento(sent))

    df_hechos[["indice_positivo", "indice_negativo", "indice_neutro"]] = (
        df_hechos["id_evaluacion"].apply(_get_indices)
    )
    df_hechos["categoria_brecha"] = df_hechos["id_evaluacion"].map(mapa_brecha)

    df_hechos.to_csv(ruta_hechos, index=False, encoding="utf-8")
    log.info("[NLP] fact_hechos actualizado con sentimiento y brecha ✔")

    # ── 6. Reporte de distribución (solo comentarios reales) ──────────────────
    df_eval_validos = df_eval.dropna(subset=["sentimiento"])
    log.info("[NLP] Distribución sentimientos (sobre %d comentarios válidos):\n%s",
             len(df_eval_validos),
             df_eval_validos["sentimiento"].value_counts().to_string())
    log.info("[NLP] Distribución brechas:\n%s",
             df_eval_validos["categoria_brecha"].value_counts().to_string())

    n_sin_eval = df_hechos["id_evaluacion"].isna().sum()
    if n_sin_eval > 0:
        log.warning("[NLP] %d filas en fact_hechos sin evaluación asociada "
                    "(bidireccionalidad=0 en esas filas)", n_sin_eval)

    log.info("[COMENTARIOS] Completado ✔ — dim_evaluacion_empresa y fact_hechos actualizados")


if __name__ == "__main__":
    main()