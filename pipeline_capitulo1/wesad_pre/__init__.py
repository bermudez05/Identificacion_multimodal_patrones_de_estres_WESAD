"""
wesad_pre — Preprocesamiento de WESAD para el trabajo de grado
"Identificación multimodal de patrones de estrés a partir de señales fisiológicas".

Módulos (en el orden del pipeline):
  config        -> todos los parámetros y su justificación
  carga         -> etapa 0: lectura de .pkl, cuestionarios y notas
  etiquetas     -> etapa 1: etiquetas y bloques de condición
  calidad       -> etapa 2: control de calidad (solo detecta)
  limpieza      -> etapas 3 y 5: filtros y huecos
  eventos       -> etapa 4: picos R, SCL/SCR, respiraciones
  procesamiento -> orquesta las etapas 2-5 por sujeto y guarda resultados
  marcadores    -> resumen fisiológico por sujeto y condición (EDA)
  estadistica   -> pruebas pareadas no paramétricas (EDA)
  ventanas      -> etapas 6 y 7: ventanas y normalización
  graficas      -> figuras compartidas
"""
from . import (calidad, carga, config, estadistica, etiquetas, eventos, graficas,
               limpieza, marcadores, procesamiento, ventanas)

__all__ = ["calidad", "carga", "config", "estadistica", "etiquetas", "eventos", "graficas",
           "limpieza", "marcadores", "procesamiento", "ventanas"]
