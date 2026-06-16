
import os
import math
import logging
from typing import Dict, List, Tuple

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

DB_CONFIG = {
    "host":     "localhost",
    "database": "vcm_practicas",
    "user":     "eciem_user",
    "password": "131313",
    "port":     "5433",
}

DATA_DIR   = os.path.join("data", "transformado")
BATCH_SIZE = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


# ─── DDL ──────────────────────────────────────────────────────────────────────

DDL_TABLAS: List[Tuple[str, str]] = [

    ("dim_rubro", """
        CREATE TABLE IF NOT EXISTS public.dim_rubro (
            id_rubro          SERIAL PRIMARY KEY,
            descripcion_rubro VARCHAR(255) NOT NULL UNIQUE
        )
    """),

    ("dim_empresa", """
        CREATE TABLE IF NOT EXISTS public.dim_empresa (
            id_empresa          SERIAL PRIMARY KEY,
            nombre              VARCHAR(255) NOT NULL,
            localidad           VARCHAR(100) NOT NULL DEFAULT 'Sin información',
            latitud             NUMERIC(9,6),
            longitud            NUMERIC(9,6),
            sectores            VARCHAR(50)  NOT NULL DEFAULT 'Sin información',
            disposicion_recibir SMALLINT     NOT NULL DEFAULT 1,
            id_rubro            INTEGER REFERENCES public.dim_rubro(id_rubro)
        )
    """),

    ("dim_carrera", """
        CREATE TABLE IF NOT EXISTS public.dim_carrera (
            id_carrera     SERIAL PRIMARY KEY,
            nombre_carrera VARCHAR(255) NOT NULL UNIQUE
        )
    """),

    ("dim_estudiante", """
        CREATE TABLE IF NOT EXISTS public.dim_estudiante (
            id_estudiante SERIAL PRIMARY KEY,
            rut_alumno    VARCHAR(20)  NOT NULL UNIQUE,
            nombre        VARCHAR(255),
            apellido      VARCHAR(255),
            correo        VARCHAR(255),
            id_carrera    INTEGER REFERENCES public.dim_carrera(id_carrera)
        )
    """),

    ("dim_tiempo", """
        CREATE TABLE IF NOT EXISTS public.dim_tiempo (
            id_tiempo  SERIAL PRIMARY KEY,
            fecha      DATE        NOT NULL UNIQUE,
            mes        INTEGER     NOT NULL,
            nombre_mes VARCHAR(20) NOT NULL,
            semestre   INTEGER     NOT NULL,
            año        INTEGER     NOT NULL
        )
    """),

    ("dim_evaluacion_empresa", """
        CREATE TABLE IF NOT EXISTS public.dim_evaluacion_empresa (
            id_evaluacion      SERIAL PRIMARY KEY,
            calificacion       NUMERIC(4,2),
            comentario         TEXT,
            sentimiento        VARCHAR(20),
            score_sentimiento  NUMERIC(5,4),
            categoria_brecha   VARCHAR(50)
        )
    """),

    ("dim_practica", """
        CREATE TABLE IF NOT EXISTS public.dim_practica (
            id_practica     SERIAL PRIMARY KEY,
            id_evaluacion   INTEGER REFERENCES public.dim_evaluacion_empresa(id_evaluacion),
            estado_practica VARCHAR(50)
        )
    """),

    ("fact_hechos", """
        CREATE TABLE IF NOT EXISTS public.fact_hechos (
            id_hechos            SERIAL PRIMARY KEY,
            id_practica          INTEGER REFERENCES public.dim_practica(id_practica),
            id_estudiante        INTEGER REFERENCES public.dim_estudiante(id_estudiante),
            id_tiempo            INTEGER REFERENCES public.dim_tiempo(id_tiempo),
            id_empresa           INTEGER REFERENCES public.dim_empresa(id_empresa),
            pertinencia          INTEGER,
            sector_publico       INTEGER,
            aprobacion           INTEGER,
            bidireccionalidad    INTEGER,
            practica_repetida    INTEGER,
            calificacion_empresa NUMERIC(4,2),
            nivel_estudio        INTEGER,
            periodicidad         VARCHAR(10),
            indice_positivo      INTEGER,
            indice_negativo      INTEGER,
            indice_neutro        INTEGER,
            categoria_brecha     VARCHAR(50),
            r_2_1   INTEGER, r_2_2  INTEGER, r_2_3  INTEGER,
            r_2_4   INTEGER, r_2_5  INTEGER, r_2_6  INTEGER,
            r_2_8   INTEGER, r_2_9  INTEGER, r_2_10 INTEGER
        )
    """),
]

TABLAS_DROP = [t[0] for t in reversed(DDL_TABLAS)]


# ─── Cargador ─────────────────────────────────────────────────────────────────

class Cargador:

    def __init__(self, db_config: Dict):
        self.conn      = None
        self.cur       = None
        self.db_config = db_config

    def conectar(self):
        self.conn = psycopg2.connect(**self.db_config)
        self.cur  = self.conn.cursor()
        log.info("[DB] Conexión establecida ✔")

    def desconectar(self):
        if self.cur:  self.cur.close()
        if self.conn: self.conn.close()
        log.info("[DB] Conexión cerrada ✔")

    def limpiar_datos(self):
        """Vacía datos sin eliminar estructura (respeta la regla de la profesora)."""
        log.info("[DB] Limpiando datos existentes...")
        for tabla in TABLAS_DROP:
            self.cur.execute(f"TRUNCATE TABLE public.{tabla} RESTART IDENTITY CASCADE")
            log.info("  → %s vaciada", tabla)
        self.conn.commit()

    def eliminar_tablas(self):
        log.info("[DB] Eliminando tablas existentes...")
        for tabla in TABLAS_DROP:
            self.cur.execute(f"DROP TABLE IF EXISTS public.{tabla} CASCADE")
            log.info("  → %s eliminada", tabla)
        self.conn.commit()

    def crear_tablas(self):
        log.info("[DB] Creando tablas...")
        for nombre, ddl in DDL_TABLAS:
            self.cur.execute(ddl)
            log.info("  → %s creada", nombre)
        self.conn.commit()

    def migrar_esquema(self):
        """
        Aplica ALTER TABLE para columnas nuevas o renombradas que no existían
        en versiones anteriores. Seguro de ejecutar múltiples veces (IF NOT EXISTS).
        """
        migraciones = [
            # sector_publico (SMALLINT) → sectores (VARCHAR) en dim_empresa
            """
            DO $$
            BEGIN
                -- Agregar 'sectores' si no existe
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'dim_empresa'
                      AND column_name  = 'sectores'
                ) THEN
                    ALTER TABLE public.dim_empresa
                        ADD COLUMN sectores VARCHAR(50) NOT NULL DEFAULT 'Sin información';
                    RAISE NOTICE 'dim_empresa: columna sectores agregada';
                END IF;

                -- Eliminar 'sector_publico' si aún existe (columna antigua)
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'dim_empresa'
                      AND column_name  = 'sector_publico'
                ) THEN
                    ALTER TABLE public.dim_empresa DROP COLUMN sector_publico;
                    RAISE NOTICE 'dim_empresa: columna sector_publico eliminada';
                END IF;

                -- Eliminar 'sector' (texto redundante) si aún existe
                IF EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'dim_empresa'
                      AND column_name  = 'sector'
                ) THEN
                    ALTER TABLE public.dim_empresa DROP COLUMN sector;
                    RAISE NOTICE 'dim_empresa: columna sector eliminada';
                END IF;

                -- Agregar 'localidad' si no existe
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'dim_empresa'
                      AND column_name  = 'localidad'
                ) THEN
                    ALTER TABLE public.dim_empresa
                        ADD COLUMN localidad VARCHAR(100) NOT NULL DEFAULT 'Sin información';
                    RAISE NOTICE 'dim_empresa: columna localidad agregada';
                END IF;

                -- Agregar 'latitud' si no existe
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'dim_empresa'
                      AND column_name  = 'latitud'
                ) THEN
                    ALTER TABLE public.dim_empresa
                        ADD COLUMN latitud NUMERIC(9,6);
                    RAISE NOTICE 'dim_empresa: columna latitud agregada';
                END IF;

                -- Agregar 'longitud' si no existe
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name   = 'dim_empresa'
                      AND column_name  = 'longitud'
                ) THEN
                    ALTER TABLE public.dim_empresa
                        ADD COLUMN longitud NUMERIC(9,6);
                    RAISE NOTICE 'dim_empresa: columna longitud agregada';
                END IF;
            END $$;
            """,
        ]
        log.info("[DB] Aplicando migraciones de esquema...")
        for sql in migraciones:
            self.cur.execute(sql)
        self.conn.commit()
        log.info("[DB] Migraciones aplicadas ✔")

    def _leer_csv(self, nombre: str) -> pd.DataFrame:
        ruta = os.path.join(DATA_DIR, f"{nombre}.csv")
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"CSV no encontrado: {ruta}")
        df = pd.read_csv(ruta, low_memory=False)
        # Convertir booleanos
        for col in df.columns:
            if df[col].dtype == bool:
                df[col] = df[col].astype(int)
        # Convertir float .0 → int donde aplique
        for col in df.columns:
            if df[col].dtype == float:
                non_null = df[col].dropna()
                if len(non_null) > 0 and (non_null == non_null.astype(int)).all():
                    df[col] = df[col].apply(lambda x: int(x) if pd.notnull(x) else None)
        df = df.replace([float('inf'), float('-inf')], None)
        df = df.where(pd.notnull(df), None)
        return df

    def _cargar_tabla(self, tabla_db: str, csv_nombre: str, columnas: List[str], pk: str = None):
        df   = self._leer_csv(csv_nombre)
        cols = [c for c in columnas if c in df.columns]
        # Convertir float NaN → None y float .0 → int para columnas numéricas
        for col in cols:
            if df[col].dtype == float:
                df[col] = df[col].apply(
                    lambda x: None if (x is None or (isinstance(x, float) and math.isnan(x)))
                    else int(x) if x == int(x) else x
                )
        datos = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
        placeholders = ", ".join(["%s"] * len(cols))

        if pk and pk in cols:
            # UPSERT: insertar o actualizar si ya existe
            update_cols = [c for c in cols if c != pk]
            update_set  = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
            query = (
                f"INSERT INTO public.{tabla_db} ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT ({pk}) DO UPDATE SET {update_set}"
            )
        else:
            # Sin PK conocida: solo insertar si no existe
            query = (
                f"INSERT INTO public.{tabla_db} ({', '.join(cols)}) VALUES ({placeholders}) "
                f"ON CONFLICT DO NOTHING"
            )

        execute_batch(self.cur, query, datos, page_size=BATCH_SIZE)
        self.conn.commit()
        log.info("[DB] %s: %d filas procesadas ✔", tabla_db, len(datos))

    def cargar_dim_rubro(self):
        self._cargar_tabla("dim_rubro", "dim_rubro", ["id_rubro", "descripcion_rubro"], pk="descripcion_rubro")

    def cargar_dim_empresa(self):
        self._cargar_tabla("dim_empresa", "dim_empresa",
                           ["id_empresa", "nombre", "localidad", "latitud", "longitud",
                            "sectores", "disposicion_recibir", "id_rubro"], pk="nombre")

    def cargar_dim_carrera(self):
        self._cargar_tabla("dim_carrera", "dim_carrera", ["id_carrera", "nombre_carrera"], pk="nombre_carrera")

    def cargar_dim_estudiante(self):
        self._cargar_tabla("dim_estudiante", "dim_estudiante",
                           ["id_estudiante", "rut_alumno", "nombre", "apellido", "correo", "id_carrera"], pk="rut_alumno")

    def cargar_dim_tiempo(self):
        self._cargar_tabla("dim_tiempo", "dim_tiempo",
                           ["id_tiempo", "fecha", "mes", "nombre_mes", "semestre", "año"], pk="fecha")

    def cargar_dim_evaluacion(self):
        """
        Carga dim_evaluacion_empresa descartando los registros cuyo sentimiento
        es NULL: son comentarios vacíos o que el modelo NLP no pudo clasificar.
        Según el criterio de la profesora, estos blancos se descartan en la carga
        para no generar espacios en blanco en los gráficos de sentimiento.
        """
        ruta = os.path.join(DATA_DIR, "dim_evaluacion_empresa.csv")
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"CSV no encontrado: {ruta}")
        df = pd.read_csv(ruta, low_memory=False)

        n_total = len(df)

        # Normalizar: convertir strings "None"/"nan" a NaN real antes de filtrar
        df["sentimiento"] = df["sentimiento"].replace({"None": None, "nan": None, "": None})
        df = df.dropna(subset=["sentimiento"])
        n_descartados = n_total - len(df)
        if n_descartados > 0:
            log.warning(
                "[CARGA] dim_evaluacion_empresa: %d registros descartados por "
                "sentimiento=NULL (blancos NLP eliminados en fase de carga)",
                n_descartados,
            )
        log.info("[CARGA] dim_evaluacion_empresa: %d registros con sentimiento válido", len(df))

        # Insertar directamente desde el DataFrame filtrado sin pasar por _cargar_tabla
        cols = ["id_evaluacion", "calificacion", "comentario", "sentimiento",
                "score_sentimiento", "categoria_brecha"]
        cols = [c for c in cols if c in df.columns]
        datos = [tuple(row) for row in df[cols].itertuples(index=False, name=None)]
        placeholders = ", ".join(["%s"] * len(cols))
        update_cols = [c for c in cols if c != "id_evaluacion"]
        update_set  = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
        query = (
            f"INSERT INTO public.dim_evaluacion_empresa ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (id_evaluacion) DO UPDATE SET {update_set}"
        )
        execute_batch(self.cur, query, datos, page_size=BATCH_SIZE)
        self.conn.commit()
        log.info("[DB] dim_evaluacion_empresa: %d filas procesadas ✔", len(datos))

    def cargar_dim_practica(self):
        df = self._leer_csv("dim_practica")

        # Obtener los id_evaluacion que realmente existen en la BD después del
        # filtro de sentimiento. Los que apuntan a evaluaciones descartadas
        # (comentario vacío o NLP fallido) se convierten en NULL para no romper
        # la FK. La práctica existió; simplemente no tiene evaluación válida cargada.
        self.cur.execute("SELECT id_evaluacion FROM public.dim_evaluacion_empresa")
        ids_validos = {row[0] for row in self.cur.fetchall()}

        n_huerfanas = 0
        datos = []
        for _, row in df.iterrows():
            id_eval = row.get("id_evaluacion")
            if id_eval is None or (isinstance(id_eval, float) and math.isnan(id_eval)):
                id_eval = None
            else:
                id_eval = int(id_eval)
                if id_eval not in ids_validos:
                    id_eval = None
                    n_huerfanas += 1
            datos.append((int(row["id_practica"]), id_eval, row["estado_practica"]))

        if n_huerfanas > 0:
            log.warning(
                "[CARGA] dim_practica: %d fila(s) con id_evaluacion anulado "
                "porque la evaluación fue descartada por sentimiento=NULL",
                n_huerfanas,
            )

        query = """
            INSERT INTO public.dim_practica (id_practica, id_evaluacion, estado_practica)
            VALUES (%s, %s, %s)
            ON CONFLICT (id_practica) DO UPDATE
            SET id_evaluacion = EXCLUDED.id_evaluacion,
                estado_practica = EXCLUDED.estado_practica
        """
        execute_batch(self.cur, query, datos, page_size=BATCH_SIZE)
        self.conn.commit()
        log.info("[DB] dim_practica: %d filas procesadas ✔", len(datos))

    def cargar_fact_hechos(self):
        import math
        df = self._leer_csv("fact_hechos")
        cols = ["id_practica", "id_estudiante", "id_tiempo", "id_empresa",
                "pertinencia", "sector_publico", "aprobacion",
                "bidireccionalidad", "practica_repetida",
                "calificacion_empresa", "nivel_estudio", "periodicidad",
                "indice_positivo", "indice_negativo", "indice_neutro",
                "categoria_brecha",
                "r_2_1", "r_2_2", "r_2_3", "r_2_4", "r_2_5",
                "r_2_6", "r_2_8", "r_2_9", "r_2_10"]
        cols = [c for c in cols if c in df.columns]

        # Columnas que deben ser INTEGER o None
        cols_int = ["id_practica", "id_estudiante", "id_tiempo", "id_empresa",
                    "pertinencia", "sector_publico", "aprobacion",
                    "bidireccionalidad", "practica_repetida", "nivel_estudio",
                    "indice_positivo", "indice_negativo", "indice_neutro",
                    "r_2_1", "r_2_2", "r_2_3", "r_2_4", "r_2_5",
                    "r_2_6", "r_2_8", "r_2_9", "r_2_10"]

        def safe_int(x):
            if x is None:
                return None
            if isinstance(x, float) and math.isnan(x):
                return None
            try:
                return int(x)
            except (ValueError, TypeError):
                return None

        datos = []
        for _, row in df[cols].iterrows():
            fila = []
            for col in cols:
                val = row[col]
                if col in cols_int:
                    fila.append(safe_int(val))
                elif val is None or (isinstance(val, float) and math.isnan(val)):
                    fila.append(None)
                else:
                    fila.append(val)
            datos.append(tuple(fila))

        placeholders = ", ".join(["%s"] * len(cols))
        update_cols = [col for col in cols if col != "id_hechos"]
        update_set  = ", ".join([f"{col} = EXCLUDED.{col}" for col in update_cols])
        query = (
            f"INSERT INTO public.fact_hechos ({', '.join(cols)}) VALUES ({placeholders}) "
            f"ON CONFLICT (id_hechos) DO UPDATE SET {update_set}"
        )
        execute_batch(self.cur, query, datos, page_size=BATCH_SIZE)
        self.conn.commit()
        log.info("[DB] fact_hechos: %d filas cargadas ✔", len(datos))

    def verificar_carga(self):
        log.info("[DB] Verificando carga...")
        for nombre, _ in DDL_TABLAS:
            self.cur.execute(f"SELECT COUNT(*) FROM public.{nombre}")
            n = self.cur.fetchone()[0]
            log.info("  %-35s → %d filas", nombre, n)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cargador = Cargador(DB_CONFIG)
    try:
        cargador.conectar()
        cargador.crear_tablas()
        cargador.migrar_esquema()
        cargador.limpiar_datos()

        log.info("[DB] Cargando dimensiones...")
        cargador.cargar_dim_rubro()
        cargador.cargar_dim_empresa()
        cargador.cargar_dim_carrera()
        cargador.cargar_dim_estudiante()
        cargador.cargar_dim_tiempo()
        cargador.cargar_dim_evaluacion()
        cargador.cargar_dim_practica()

        log.info("[DB] Cargando tabla de hechos...")
        cargador.cargar_fact_hechos()

        cargador.verificar_carga()
        log.info("[DB] Carga completada exitosamente ✔")

    except Exception as e:
        log.error("[DB] Error: %s", e)
        raise
    finally:
        cargador.desconectar()

if __name__ == "__main__":
    main()