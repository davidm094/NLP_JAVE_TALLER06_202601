"""Medición simple de la calidad del texto generado según la estrategia de muestreo.

Para cada estrategia (argmax y T = 0,5 / 0,8 / 1,0) se generan 6 textos de 500 caracteres
(2 semillas x 3 prompts) y se mide:
  - palabras reales: % de palabras generadas que aparecen en el libro;
  - bigramas distintos: bigramas de palabras distintos / total (1 = nada se repite);
  - copia literal: % de 5-gramas de palabras generados que aparecen tal cual en el libro
    (memorización), y el tramo más largo de palabras seguidas copiado del libro.
"""
import json, re, sys
from pathlib import Path

import pandas as pd
import keras

sys.path.insert(0, str(Path(__file__).parent))
from libro import cargar_libro, PROMPTS
from generacion import preparar, modelo_generador, generar

RAIZ = Path(__file__).resolve().parents[1]
OCULTO = int(sys.argv[1]) if len(sys.argv) > 1 else 512
palabras = lambda t: re.findall(r"[a-záéíóúüñ]+", t.lower())

texto = cargar_libro()
lexico = set(palabras(texto))
wl = palabras(texto)
cinco = {tuple(wl[i:i + 5]) for i in range(len(wl) - 4)}
libro_plano = " " + " ".join(wl) + " "


def tramo_copiado_mas_largo(w):
    """Mayor número de palabras seguidas del texto generado que aparecen así en el libro."""
    mejor = 0
    for i in range(len(w)):
        j = i + mejor + 1
        while j <= len(w) and (" " + " ".join(w[i:j]) + " ") in libro_plano:
            mejor = j - i
            j += 1
    return mejor
tok, _ = preparar(texto)
gen = modelo_generador(keras.models.load_model(RAIZ / "results" / f"generacion_gru{OCULTO}.keras"), len(tok), oculto=OCULTO)

prompts = PROMPTS
filas = []
for T in [None, 0.5, 0.8, 1.0]:
    reales = total = 0
    bigr_dist = bigr_tot = 0
    c5_copiados = c5_tot = 0
    tramo = 0
    for s in [101, 202]:
        for k, p in enumerate(prompts):
            if T is None and s == 202:
                continue  # argmax es determinista: la segunda semilla daría lo mismo
            w = palabras(generar(gen, tok, p, largo=500, temperatura=T, oculto=OCULTO, semilla=s + k)[len(p):])
            reales += sum(x in lexico for x in w); total += len(w)
            b = list(zip(w, w[1:])); bigr_dist += len(set(b)); bigr_tot += len(b)
            g5 = [tuple(w[i:i + 5]) for i in range(len(w) - 4)]
            c5_copiados += sum(g in cinco for g in g5); c5_tot += len(g5)
            tramo = max(tramo, tramo_copiado_mas_largo(w))
    filas.append(dict(estrategia="argmax" if T is None else f"T={T}", palabras=total,
                      pct_palabras_reales=100 * reales / total, bigramas_distintos=bigr_dist / bigr_tot,
                      pct_5gramas_copiados=100 * c5_copiados / c5_tot, tramo_copiado_max=tramo))
    print(filas[-1], flush=True)
pd.DataFrame(filas).to_csv(RAIZ / "results" / "calidad_generacion.csv", index=False)
