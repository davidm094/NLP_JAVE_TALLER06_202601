"""Entrena el modelo de lenguaje por caracteres y guarda resultados en results/."""
import json, sys, time
from pathlib import Path

import numpy as np
import keras

sys.path.insert(0, str(Path(__file__).parent))
from libro import cargar_libro, TITULO, AUTOR, PROMPTS
from generacion import preparar, construir_modelo, modelo_generador, generar

RAIZ = Path(__file__).resolve().parents[1]
OUT = RAIZ / "results"
OUT.mkdir(exist_ok=True)

OCULTO = int(sys.argv[1]) if len(sys.argv) > 1 else 512
EPOCAS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
keras.utils.set_random_seed(42)

texto = cargar_libro()
tok, d = preparar(texto)
modelo = construir_modelo(len(tok), oculto=OCULTO)
t0 = time.time()
hist = modelo.fit(
    d["x_tr"], d["y_tr"], batch_size=64, epochs=EPOCAS,
    validation_data=(d["x_va"], d["y_va"]),
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
    verbose=2,
)
minutos = (time.time() - t0) / 60
modelo.save(OUT / f"generacion_gru{OCULTO}.keras")

val_loss, val_acc = modelo.evaluate(d["x_va"], d["y_va"], verbose=0)
mejor = int(np.argmin(hist.history["val_loss"]))
resumen = dict(
    libro=f"{TITULO} — {AUTOR}",
    caracteres=len(texto), vocab=len(tok), secuencias_train=len(d["x_tr"]), secuencias_val=len(d["x_va"]),
    oculto=OCULTO, epocas_corridas=len(hist.history["loss"]), mejor_epoca=mejor + 1,
    val_loss=val_loss, val_acc=val_acc, perplejidad_val=float(np.exp(val_loss)),
    train_loss_mejor=hist.history["loss"][mejor], train_acc_mejor=hist.history["sparse_categorical_accuracy"][mejor],
    minutos=minutos,
)
json.dump(hist.history, open(OUT / "historia_generacion.json", "w"), indent=1)

gen = modelo_generador(modelo, len(tok), oculto=OCULTO)
prompts = PROMPTS
muestras = []
for k, p in enumerate(prompts):
    for T in [None, 0.5, 0.8, 1.0]:
        muestras.append(dict(prompt=p, temperatura="voraz" if T is None else T,
                             texto=generar(gen, tok, p, largo=350, temperatura=T, oculto=OCULTO, semilla=7 + k)))
        print("\n===", p, T, "===\n", muestras[-1]["texto"], flush=True)
json.dump(muestras, open(OUT / "muestras_generacion.json", "w"), ensure_ascii=False, indent=1)
json.dump(resumen, open(OUT / "resumen_generacion.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(resumen, indent=1, ensure_ascii=False))
