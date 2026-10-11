# Pipeline del capítulo 1 — EDA y preprocesamiento de WESAD

## Cómo usarlo

1. Esta carpeta (`pipeline_capitulo1/`) debe estar al lado de la carpeta `WESAD/` (con `S2/`, `S3/`, …).
2. Instala las librerías una sola vez, en el entorno de Anaconda que usas:
   ```
   pip install neurokit2 cvxopt seaborn
   ```
3. Abre y ejecuta los notebooks **en orden**. Cada uno explica qué hace y cómo leer sus gráficas.

| Notebook | Qué hace | Tiempo aprox. |
|---|---|---|
| `01_inventario_y_etiquetas` | Revisa que los datos estén completos, las condiciones del protocolo, las 10 señales y los cuestionarios | 1–5 min |
| `02_control_de_calidad` | Busca muestras dañadas en las 10 señales | 2–5 min |
| `03_limpieza_y_eventos` | Limpia las 7 señales del modelo y detecta latidos, respiraciones y respuestas de sudoración | 20–30 min |
| `04_marcadores_y_variabilidad` | Muestra si el estrés se ve en cada señal (objetivo 1) | < 1 min |
| `05_ventanas_y_exportacion` | Corta en ventanas de 60 s con sus etiquetas y exporta para el objetivo 2 | < 1 min |

Los notebooks ya vienen ejecutados con los datos reales, así que puedes leerlos sin correrlos.

## Qué hay en cada carpeta

- `wesad_pre/` — el código del pipeline. **Todos los parámetros están en `wesad_pre/config.py`**, con su justificación.
- `salidas/tablas/` — tablas para el documento (calidad, eventos, marcadores, pruebas estadísticas, ventanas).
- `salidas/figuras/` — todas las figuras en PNG, listas para el documento. Las de sesión completa (`01_sesion_completa_S*.png`)
  y antes/después de la limpieza (`03_antes_despues_S*.png`) están para cada uno de los 15 sujetos.
- `salidas/qc/` — máscaras de calidad de los 15 sujetos (un `.npz` por sujeto; las crea el notebook 02).
- `salidas/limpias/` — señales limpias y eventos de los 15 sujetos (un `.pkl.gz` por sujeto; las crea el notebook 03).
  Ya vienen incluidas, así que los notebooks 04 y 05 se pueden correr sin repetir el 03.

## Señales

- **Se limpian y van al modelo:** ECG, EDA de pecho, respiración, BVP, EDA de muñeca, temperatura de pecho y de muñeca.
- **Solo EDA (análisis exploratorio):** EMG y los dos acelerómetros. Los acelerómetros sirven para medir movimiento.

## Importante

Los `.pkl` de WESAD **ya vienen en unidades físicas**.
