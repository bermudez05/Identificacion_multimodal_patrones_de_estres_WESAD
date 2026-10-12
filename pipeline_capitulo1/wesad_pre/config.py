"""
config.py — Toda la configuración del pipeline en un solo lugar.

Cada número que aparece aquí tiene una justificación en la literatura (columna
"Fuente" de la tabla del documento del capítulo 1). Si el equipo decide cambiar
algo, se cambia aquí y todos los notebooks lo toman automáticamente.

Convención de nombres: cada señal se identifica con "<dispositivo>_<canal>",
por ejemplo "chest_ECG" (pecho, RespiBAN) o "wrist_EDA" (muñeca, Empatica E4).
"""

# ---------------------------------------------------------------------------
# 1. Datos generales del dataset
# ---------------------------------------------------------------------------

# S1 y S12 fueron descartados por los autores de WESAD por falla del sensor.
SUBJECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
# (Solo para pruebas rápidas: la variable de entorno WESAD_SUBJECTS="2,3" limita los sujetos.)
import os as _os
if _os.environ.get("WESAD_SUBJECTS"):
    SUBJECT_IDS = [int(s) for s in _os.environ["WESAD_SUBJECTS"].split(",")]

# Las etiquetas vienen a 700 Hz, sincronizadas con el RespiBAN (pecho).
FS_LABEL = 700

# Códigos de condición del protocolo. 0 = transición, 5-7 = ignorar (según el readme de WESAD).
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
COND_ORDER = ["baseline", "stress", "amusement", "meditation"]

# La red eléctrica de Alemania (donde se grabó WESAD) es de 50 Hz, no 60 Hz.
POWERLINE_HZ = 50

# ---------------------------------------------------------------------------
# 2. Catálogo de señales
# ---------------------------------------------------------------------------
# rol:
#   "nucleo"      -> se limpia y se usa para el modelo (objetivo 2)
#   "apoyo"       -> se limpia y se usa, pero se espera menor aporte (temperatura)
#   "movimiento"  -> NO es predictor; sirve para marcar artefactos de movimiento
#   "exploratoria"-> solo se analiza en el EDA, no se limpia para el modelo (EMG y BVP)
#
# Los .pkl de WESAD ya traen el pecho en unidades físicas (mV, µS, °C, %, g).
# Lo único que hay que convertir es el ACC de muñeca: el E4 entrega 1/64 g.

SIGNALS = {
    "chest_ECG":  dict(device="chest", channel="ECG",  fs=700, unit="mV", role="nucleo",       name="ECG (pecho)"),
    "chest_EDA":  dict(device="chest", channel="EDA",  fs=700, unit="µS", role="nucleo",       name="EDA (pecho)"),
    "chest_Resp": dict(device="chest", channel="Resp", fs=700, unit="%",  role="nucleo",       name="Respiración (pecho)"),
    # BVP: el grupo decidió no usarlo en el modelo (oct. 2026). Queda solo para el EDA, como el EMG.
    "wrist_BVP":  dict(device="wrist", channel="BVP",  fs=64,  unit="u.a.", role="exploratoria", name="BVP (muñeca)"),
    "wrist_EDA":  dict(device="wrist", channel="EDA",  fs=4,   unit="µS", role="nucleo",       name="EDA (muñeca)"),
    "chest_Temp": dict(device="chest", channel="Temp", fs=700, unit="°C", role="apoyo",        name="Temperatura (pecho)"),
    "wrist_TEMP": dict(device="wrist", channel="TEMP", fs=4,   unit="°C", role="apoyo",        name="Temperatura (muñeca)"),
    "chest_ACC":  dict(device="chest", channel="ACC",  fs=700, unit="g",  role="movimiento",   name="Acelerómetro (pecho)"),
    "wrist_ACC":  dict(device="wrist", channel="ACC",  fs=32,  unit="g",  role="movimiento",   name="Acelerómetro (muñeca)"),
    "chest_EMG":  dict(device="chest", channel="EMG",  fs=700, unit="mV", role="exploratoria", name="EMG (pecho)"),
}

ALL_SIGNALS = list(SIGNALS)
# Señales que se limpian y salen hacia el objetivo 2
MODEL_SIGNALS = [k for k, v in SIGNALS.items() if v["role"] in ("nucleo", "apoyo")]
MOTION_SIGNALS = [k for k, v in SIGNALS.items() if v["role"] == "movimiento"]

# ---------------------------------------------------------------------------
# 3. Control de calidad (etapa 2) — SOLO DETECTA, no modifica la señal
# ---------------------------------------------------------------------------
# Cada criterio tiene un "nivel":
#   "excluir" -> la muestra no es fisiológica o el sensor no estaba midiendo
#   "marcar"  -> la muestra se conserva pero queda señalada
#
# range        : (mín, máx) fuera de lo cual el valor es imposible.
# rail         : |x| >= rail sostenido min_s segundos = sensor saturado.
# flat_std     : desviación estándar móvil < umbral durante >= min_s = señal plana.
# flat_equal_s : el MISMO valor repetido durante >= este tiempo = sensor pegado
#                (para señales lentas como EDA y temperatura).
# slope        : cambio más rápido que este valor por segundo (Kleckner et al. 2018).

QC = {
    # rail_breve: toques cortos del tope (< min_s de "rail"), p. ej. la punta de una onda R
    # recortada. No invalidan la señal, pero se marcan para poder revisarlos.
    "chest_ECG": dict(
        range=(-5.0, 5.0),
        rail=dict(value=1.49, min_s=0.1, level="excluir"),
        rail_breve=dict(value=1.49, level="marcar"),
        flat_std=dict(value=0.005, min_s=2.0, level="excluir"),
    ),
    "chest_EMG": dict(
        range=(-5.0, 5.0),
        rail=dict(value=1.49, min_s=0.1, level="excluir"),
        rail_breve=dict(value=1.49, level="marcar"),
        flat_std=dict(value=0.0005, min_s=2.0, level="excluir"),
    ),
    "chest_EDA": dict(
        range=(0.05, 60.0),                       # Kleckner et al. 2018
        flat_equal_s=dict(value=60.0, level="excluir"),
        slope=dict(value=10.0, pad_s=5.0, level="excluir"),   # µS/s, ±5 s alrededor (Kleckner)
    ),
    "wrist_EDA": dict(
        range=(0.05, 60.0),
        flat_equal_s=dict(value=60.0, level="excluir"),
        slope=dict(value=10.0, pad_s=5.0, level="excluir"),
    ),
    "chest_Resp": dict(
        range=(-60.0, 60.0),
        rail=dict(value=49.9, min_s=0.5, level="excluir"),
        flat_std=dict(value=0.05, min_s=10.0, level="excluir"),
    ),
    "chest_Temp": dict(
        range=(20.0, 42.0),                       # incluye el centinela -273.15 °C
        flat_equal_s=dict(value=60.0, level="excluir"),
    ),
    "wrist_TEMP": dict(
        range=(20.0, 42.0),
        flat_equal_s=dict(value=60.0, level="excluir"),
    ),
    "wrist_BVP": dict(
        flat_std=dict(value=0.01, min_s=2.0, level="excluir"),
    ),
    "chest_ACC": dict(
        range=(-8.0, 8.0),
    ),
    "wrist_ACC": dict(
        range=(-8.0, 8.0),
        rail=dict(value=1.98, min_s=0.0, level="marcar"),    # ±128/64 = ±2 g: saturación del E4
    ),
}

# ---------------------------------------------------------------------------
# 4. Limpieza (etapa 3) y frecuencias de trabajo
# ---------------------------------------------------------------------------
# Todos los filtros son de FASE CERO (se aplican hacia adelante y hacia atrás),
# para no desplazar en el tiempo los picos R, las SCR ni los ciclos respiratorios.

CLEAN = {
    # Pasa-altas 0.5 Hz de fase cero: Kligfield et al. 2007 piden 0.05 Hz para ECG diagnóstico y
    # admiten hasta 0.67 Hz con filtros de fase cero; aquí solo interesan los picos R.
    # Notch en la red. Pasa-bajas 40 Hz: decisión del proyecto para detectar picos R.
    "chest_ECG":  dict(notch_hz=POWERLINE_HZ, band=(0.5, 40.0), order=4, fs_out=700),
    # Schmidt et al. 2018 (WESAD): pasa-bajas 5 Hz. Luego se baja a 8 Hz (suficiente para EDA).
    "chest_EDA":  dict(lowpass=5.0, order=4, fs_out=8),
    # Empatica E4 ya muestrea a 4 Hz (Nyquist 2 Hz): pasa-bajas 1 Hz (decisión del proyecto).
    "wrist_EDA":  dict(lowpass=1.0, order=4, fs_out=4),
    # Khodadad et al. 2018 (método por defecto de NeuroKit2: pasa-banda 0.05–3 Hz). Antes se baja a 32 Hz.
    "chest_Resp": dict(method="khodadad2018", fs_out=32),
    # Temperatura: cambia muy lento; pasa-bajas 0.1 Hz y 4 Hz de trabajo.
    "chest_Temp": dict(lowpass=0.1, order=2, fs_out=4),
    "wrist_TEMP": dict(lowpass=0.1, order=2, fs_out=4),
}

# Huecos (etapa 5): solo se interpolan huecos cortos y solo en señales lentas.
MAX_GAP_INTERP_S = {"chest_EDA": 2.0, "wrist_EDA": 2.0, "chest_Temp": 2.0, "wrist_TEMP": 2.0}

# ---------------------------------------------------------------------------
# 5. Eventos y calidad fisiológica (etapa 4)
# ---------------------------------------------------------------------------
EVENTS = {
    # Rango fisiológico de intervalos RR: 30–200 lpm
    "rr_ms_range": (300, 2000),
    # Calidad de ECG: clasificación de Zhao y Zhang (2018) en segmentos de 10 s
    # (excelente / aceptable / inaceptable). Los RR en segmentos inaceptables no se usan.
    "ecg_sqi_segment_s": 10.0,
    # SCR: amplitud mínima absoluta 0.01 µS (Boucsein et al. 2012)
    "scr_min_amplitude_us": 0.01,
    # Movimiento: desviación estándar de la magnitud de ACC en 1 s
    "motion_window_s": 1.0,
}

# ---------------------------------------------------------------------------
# 6. Ventanas (etapa 6) — opción estándar de WESAD (Schmidt et al. 2018)
# ---------------------------------------------------------------------------
WINDOW = dict(
    length_s=60.0,          # 60 s para señales fisiológicas
    step_s=0.25,            # desplazamiento de 0.25 s
    min_valid_frac=0.80,    # una modalidad es válida en la ventana si >= 80 % de muestras válidas
    trim_start_s=0.0,       # segundos a descartar al inicio de cada bloque (0 = como WESAD)
)

# Tareas de clasificación que se preparan (se hacen ambas, la elección es del capítulo 2)
#   3 clases: baseline / stress / amusement (meditación fuera), como en WESAD
#   binaria : stress (1) vs. no-stress = baseline + amusement (0), como en WESAD
TASKS = {
    "y_3clases": {"baseline": 0, "stress": 1, "amusement": 2},
    "y_binaria": {"baseline": 0, "stress": 1, "amusement": 0},
}

# ---------------------------------------------------------------------------
# 7. Carpetas de salida
# ---------------------------------------------------------------------------
OUT_DIRS = dict(
    qc="salidas/qc",
    clean="salidas/limpias",
    tables="salidas/tablas",
    figures="salidas/figuras",
)
