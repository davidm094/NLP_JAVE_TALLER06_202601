"""Calcula todas las métricas de traducción a partir de las predicciones del test.

Lo usa correr_traduccion.py después de traducir, y también se puede correr solo
(python src/evaluar_traduccion.py) para recalcular las métricas desde
results/traducciones_test.csv sin volver a entrenar ni a traducir.

Genera:
  results/traducciones_test.csv   predicción y métricas por frase
  results/metricas_por_largo.csv  métricas por tramo de longitud
  results/validacion_metricas.csv implementación propia vs sacrebleu / rouge-score
  y actualiza 'metricas_test' y 'linea_base_copiar_espanol' en resumen_traduccion.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from traduccion import evaluar
from metricas import rouge_n, rouge_l

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results"


def validar(hip, ref):
    """Compara la implementación propia con las librerías de referencia."""
    import sacrebleu
    from rouge_score import rouge_scorer
    from metricas import bleu_corpus

    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    lib = {"rouge1": [], "rouge2": [], "rougeL": []}
    pro = {"rouge1": [], "rouge2": [], "rougeL": []}
    for h, r in zip(hip, ref):
        s = sc.score(r, h)
        for k in lib:
            lib[k].append(s[k].fmeasure)
        pro["rouge1"].append(rouge_n(h, r, 1)); pro["rouge2"].append(rouge_n(h, r, 2)); pro["rougeL"].append(rouge_l(h, r))
    filas = [dict(metrica="BLEU (corpus)", propia=bleu_corpus(hip, ref)[0] * 100,
                  libreria=sacrebleu.corpus_bleu(hip, [ref], tokenize="none").score,
                  libreria_nombre="sacrebleu.corpus_bleu(tokenize='none')", max_dif_por_frase=np.nan),
             dict(metrica="BLEU (corpus), tokenizador 13a", propia=np.nan,
                  libreria=sacrebleu.corpus_bleu(hip, [ref]).score,
                  libreria_nombre="sacrebleu.corpus_bleu (por defecto, 13a)", max_dif_por_frase=np.nan)]
    for k, nombre in [("rouge1", "ROUGE-1"), ("rouge2", "ROUGE-2"), ("rougeL", "ROUGE-L")]:
        a, b = np.array(pro[k]), np.array(lib[k])
        filas.append(dict(metrica=nombre + " (promedio)", propia=a.mean() * 100, libreria=b.mean() * 100,
                          libreria_nombre="rouge_score.RougeScorer", max_dif_por_frase=np.abs(a - b).max() * 100))
    d = pd.DataFrame(filas)
    d["diferencia"] = d.propia - d.libreria
    return d


def calcular_y_guardar(tabla):
    """tabla: DataFrame con columnas es, en, status, prediccion (filas del test)."""
    tabla = tabla[["es", "en", "status", "prediccion"]].copy().reset_index(drop=True)
    tabla["prediccion"] = tabla.prediccion.fillna("")
    hip, ref = tabla.prediccion.tolist(), tabla.en.tolist()
    resumen, por_frase = evaluar(hip, ref)
    base, _ = evaluar(tabla.es.tolist(), ref)  # línea base: copiar el español

    tabla["largo_es"] = tabla.es.str.split().str.len()
    tabla = pd.concat([tabla, por_frase], axis=1)
    tabla.to_csv(OUT / "traducciones_test.csv", index=False)

    tramos = pd.cut(tabla.largo_es, [0, 5, 10, 15, 20], labels=["1-5", "6-10", "11-15", "16-20"])
    por_largo = []
    for nombre, g in tabla.groupby(tramos, observed=True):
        r, _ = evaluar(g.prediccion.tolist(), g.en.tolist())
        por_largo.append(dict(tramo=str(nombre), n=len(g), BLEU=r["BLEU"], ROUGE1=r["ROUGE1"],
                              ROUGE2=r["ROUGE2"], ROUGEL=r["ROUGEL"], F1=r["F1"]))
    pd.DataFrame(por_largo).to_csv(OUT / "metricas_por_largo.csv", index=False)
    validar(hip, ref).to_csv(OUT / "validacion_metricas.csv", index=False)
    return resumen, base, pd.DataFrame(por_largo)


if __name__ == "__main__":
    t = pd.read_csv(OUT / "traducciones_test.csv", keep_default_na=False)
    resumen, base, por_largo = calcular_y_guardar(t)
    ruta = OUT / "resumen_traduccion.json"
    final = json.load(open(ruta))
    final["metricas_test"], final["linea_base_copiar_espanol"] = resumen, base
    json.dump(final, open(ruta, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({"modelo": resumen, "linea_base": base}, indent=1, ensure_ascii=False))
    print(por_largo.round(2).to_string())
    print(pd.read_csv(OUT / "validacion_metricas.csv").round(4).to_string())
