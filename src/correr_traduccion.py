"""Entrena el seq2seq español -> inglés, traduce el test y calcula BLEU, ROUGE y F1."""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import keras

sys.path.insert(0, str(Path(__file__).parent))
from traduccion import (cargar_pares, particionar, preparar, construir_seq2seq,
                        traducir_lote)
from evaluar_traduccion import calcular_y_guardar

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results"
OUT.mkdir(exist_ok=True)
DATOS = RAIZ / "data" / "Combined_Data_es-ES.csv.gz"

OCULTO = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
EPOCAS = int(sys.argv[2]) if len(sys.argv) > 2 else 15
keras.utils.set_random_seed(42)

df, conteo = cargar_pares(DATOS)
tr, va, te = particionar(df)
v_es, v_en, Dtr, Dva, Dte = preparar(tr, va, te)

modelo = construir_seq2seq(len(v_es), len(v_en), oculto=OCULTO)
t0 = time.time()
hist = modelo.fit(
    Dtr[0], Dtr[1], sample_weight=Dtr[2], batch_size=64, epochs=EPOCAS,
    validation_data=Dva,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    verbose=2,
)
minutos = (time.time() - t0) / 60
modelo.save(OUT / f"seq2seq_gru{OCULTO}.keras")
json.dump(hist.history, open(OUT / "historia_traduccion.json", "w"), indent=1)

# ---- traducción del test completo y métricas
t1 = time.time()
hip = traducir_lote(modelo, v_es, v_en, te.es.tolist())
min_dec = (time.time() - t1) / 60
tabla = te[["es", "en", "status"]].copy().reset_index(drop=True)
tabla["prediccion"] = hip
resumen, base, por_largo = calcular_y_guardar(tabla)  # métricas propias + validación

mejor = int(np.argmin(hist.history["val_loss"]))
final = dict(
    datos=conteo, train=len(tr), val=len(va), test=len(te), vocab_es=len(v_es), vocab_en=len(v_en),
    unk_test_es=float((Dte[0]["spanish"] == 1).sum() / (Dte[0]["spanish"] > 0).sum()),
    oculto=OCULTO, epocas_corridas=len(hist.history["loss"]), mejor_epoca=mejor + 1,
    val_loss=hist.history["val_loss"][mejor], val_acc=hist.history["val_accuracy"][mejor],
    minutos_entrenamiento=minutos, minutos_decodificacion=min_dec,
    metricas_test=resumen, linea_base_copiar_espanol=base,
)
json.dump(final, open(OUT / "resumen_traduccion.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(final, indent=1, ensure_ascii=False))
print(por_largo.round(2).to_string())
