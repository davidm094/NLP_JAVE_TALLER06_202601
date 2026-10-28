"""Genera notebooks/Taller06_Generacion_y_Traduccion.ipynb (luego se ejecuta con nbconvert)."""
from pathlib import Path

import nbformat as nbf

REPO = "https://github.com/davidm094/NLP_JAVE_TALLER06_202601"
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s))

md(f"""# Taller 06 — Generación de texto y traducción seq2seq

**Curso:** Procesamiento de Lenguaje Natural — Pontificia Universidad Javeriana
**Autores:** David Moreno · Dierick Brochero
**Repositorio:** {REPO}

Adaptación del notebook de clase (*Deep Learning with Python*, 3a ed., cap. 15):

1. **Generación de texto:** modelo de lenguaje por caracteres (GRU) entrenado con *Rebelión en la granja*
   (George Orwell, 1945; trad. de Marcial Souto), elegido por el grupo porque David Moreno lo leyó.
2. **Traducción español → inglés:** seq2seq con GRU (codificador bidireccional) sobre *Combined Data es-ES*,
   evaluado con **BLEU**, **ROUGE** y **F1**.

El código está en `src/`. Con `ENTRENAR = False` el notebook carga los modelos de `results/` **si existen** y recalcula
generaciones y métricas; si no existen, los entrena. En un clon del repositorio (Colab) el modelo de caracteres sí está
(`results/generacion_gru512.keras`), pero el seq2seq (351 MB) no cabe en GitHub y **se reentrena**: sus métricas pueden variar
un poco respecto al reporte. Con `ENTRENAR = True` se reentrenan ambos. Las celdas que calculan métricas sobrescriben los
archivos de `results/` con la corrida actual.

**Sobre el libro:** *Rebelión en la granja* no es de dominio público en Colombia (Orwell murió en 1950 y el plazo es la vida
del autor + 80 años, hasta 2031). Por eso el texto **no está en el repositorio**: hay que subir una copia propia en epub
(celda de carga de la Parte 1). Se usa solo para entrenar el modelo en esta actividad académica.

**En Colab:** *Entorno de ejecución → Cambiar tipo → GPU T4* y ejecutar todo (el seq2seq tarda mucho menos con GPU).
El modelo de caracteres usa una GRU de 512 (el notebook de clase usa 1024) para reproducir el reporte; se cambia con `GEN_OCULTO`.""")

code(f"""import os, sys, subprocess
from pathlib import Path
if "google.colab" in sys.modules and Path.cwd().name != "NLP_JAVE_TALLER06_202601":
    if not Path("NLP_JAVE_TALLER06_202601").exists():
        subprocess.run(["git", "clone", "-q", "{REPO}"], check=True)
    os.chdir("NLP_JAVE_TALLER06_202601")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "rouge-score"], check=True)
RAIZ = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RAIZ / "src"))
os.environ.setdefault("KERAS_BACKEND", "tensorflow")""")

code("""import json, numpy as np, pandas as pd, keras, tensorflow as tf
from IPython.display import Image, display
import figuras as F
pd.set_option("display.max_colwidth", 120)
GPU = len(tf.config.list_physical_devices("GPU")) > 0
ENTRENAR = False          # True = reentrenar los dos modelos
GEN_OCULTO = 512          # el notebook de clase usa 1024; 512 reproduce el reporte
S2S_OCULTO = 1024
print("Keras", keras.__version__, "| backend", keras.backend.backend(), "| GPU:", GPU)
keras.utils.set_random_seed(42)""")

# ------------------------------------------------------------- Parte 1
md("""## Parte 1. Modelo de lenguaje por caracteres

### 1.1 Corpus
Del epub se toman solo los capítulos I a X (la novela). Quedan fuera la portada, los créditos, el prólogo de Orwell
(*La libertad de prensa*), el epílogo y las notas. Queda un párrafo por línea.""")
code("""from libro import cargar_libro, ARCHIVO, PROMPTS
if not ARCHIVO.exists() and "google.colab" in sys.modules:
    from google.colab import files
    print("Suba su copia de Rebelión en la granja en formato epub")
    subido = files.upload()
    if not subido:
        raise RuntimeError("No se subió ningún archivo: vuelva a ejecutar esta celda y elija el epub.")
    ARCHIVO.parent.mkdir(exist_ok=True)
    ARCHIVO.write_bytes(next(iter(subido.values())))""")
code("""from generacion import preparar, construir_modelo, modelo_generador, generar
texto = cargar_libro()
print(f"{len(texto):,} caracteres, {len(set(texto))} caracteres distintos")
print(texto[:300])  # fragmento corto: el libro tiene derechos vigentes""")

md("""### 1.2 Datos de entrenamiento
Igual que en el notebook: ventanas no solapadas de 100 caracteres; la etiqueta es la misma ventana corrida un carácter.
A diferencia del notebook, el **último 10 % del libro** se reserva como validación para medir la pérdida en texto no visto
y detener el entrenamiento con *early stopping* (paciencia 3).""")
code("""tok, d = preparar(texto, largo=100, frac_val=0.10)
print("vocabulario:", len(tok), "| train:", d["x_tr"].shape, "| val:", d["x_va"].shape)
print("x:", repr(tok.decodificar(d["x_tr"][0][:60])))
print("y:", repr(tok.decodificar(d["y_tr"][0][:60])))""")

md("### 1.3 Modelo y entrenamiento")
code("""ruta_gen = RAIZ / "results" / f"generacion_gru{GEN_OCULTO}.keras"
modelo = None
if not ENTRENAR and ruta_gen.exists():
    try:
        modelo = keras.models.load_model(ruta_gen)
        print("Modelo cargado de", ruta_gen.name)
    except Exception as e:  # p. ej. otra versión de Keras: se reentrena
        print("No se pudo cargar el modelo guardado (", e, "); se reentrena.")
if modelo is None:
    modelo = construir_modelo(len(tok), oculto=GEN_OCULTO)
    hist = modelo.fit(d["x_tr"], d["y_tr"], batch_size=64, epochs=60,
                      validation_data=(d["x_va"], d["y_va"]),
                      callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                               restore_best_weights=True)])
    json.dump(hist.history, open(RAIZ / "results" / "historia_generacion.json", "w"))
    modelo.save(ruta_gen)
modelo.summary(line_length=80)
val_loss, val_acc = modelo.evaluate(d["x_va"], d["y_va"], verbose=0)
print(f"Validación: pérdida {val_loss:.3f} nats/carácter | perplejidad {np.exp(val_loss):.2f} | exactitud {val_acc:.3f}")""")
md("Curvas de la historia de entrenamiento guardada en `results/` (si se reentrenó arriba, son las de esta corrida).")
code("""F.fig_curvas_generacion()
display(Image(str(RAIZ / "figs" / "curvas_generacion.png")))""")

md("""### 1.4 Generación
El notebook de clase usa **argmax** (siempre el carácter más probable), lo que tiende a ciclos repetitivos.
Se compara con **muestreo con temperatura** *T*: se divide el logaritmo de las probabilidades por *T* y se vuelve a
normalizar. *T* < 1 concentra la distribución (texto más conservador); *T* = 1 muestrea del modelo tal cual.""")
code("""gen = modelo_generador(modelo, len(tok), oculto=GEN_OCULTO)
for k, prompt in enumerate(PROMPTS):     # una semilla por prompt, igual que en el reporte
    for T in [None, 0.5, 0.8, 1.0]:
        print(f"\\n=== prompt: {prompt!r} | {'voraz (argmax)' if T is None else f'T = {T}'} ===")
        print(generar(gen, tok, prompt, largo=350, temperatura=T, oculto=GEN_OCULTO, semilla=7 + k))""")
md("""### 1.5 Calidad del texto según la estrategia
Para no juzgar a ojo: 500 caracteres por prompt y estrategia (dos semillas con muestreo). Se mide qué porcentaje de las
palabras existe en el libro, qué fracción de bigramas es distinta (repetición), qué porcentaje de 5-gramas de palabras está
**copiado literalmente del libro** (memorización) y el tramo copiado más largo. Toma unos minutos en CPU.""")
code("""subprocess.run([sys.executable, str(RAIZ / "src" / "evaluar_generacion.py"), str(GEN_OCULTO)], check=True)
pd.read_csv(RAIZ / "results" / "calidad_generacion.csv").round(2)""")

# ------------------------------------------------------------- Parte 2
md("""## Parte 2. Traducción español → inglés (seq2seq)

### 2.1 Datos
*Combined Data es-ES* tiene la columna `statement` (inglés original) y `statement_es_es` (traducción al español).
Pasos de limpieza:

1. normalizar (minúsculas, sin puntuación ni ¿¡; el apóstrofo se borra: *don't → dont*);
2. quedarse con frases cortas: ≤ 20 palabras en español y ≤ 19 en inglés (para que `[start]` + 19 + `[end]` quepa en las
   21 posiciones del notebook). Un seq2seq sin atención comprime toda la frase en un solo vector, y la mediana del dataset
   es de 53 palabras en español y 62 en inglés;
3. quitar pares duplicados;
4. quitar pares cuyo "español" es idéntico al inglés (no están traducidos: hashes, risas, otros idiomas).

Partición aleatoria 70 / 15 / 15 (semilla 42), igual que el notebook.""")
code("""from traduccion import (cargar_pares, particionar, preparar as preparar_tr, construir_seq2seq,
                        traducir_lote, evaluar)
df, conteo = cargar_pares(RAIZ / "data" / "Combined_Data_es-ES.csv.gz")
print(conteo)
tr, va, te = particionar(df)
v_es, v_en, Dtr, Dva, Dte = preparar_tr(tr, va, te)
print(f"train {len(tr)} | val {len(va)} | test {len(te)} | vocab es {len(v_es)} | vocab en {len(v_en)}")
tr[["es", "en"]].head(8)""")

md("""### 2.2 Modelo
Arquitectura del notebook: embedding (256) → GRU bidireccional de 1024 unidades (suma de direcciones) como codificador;
su estado final inicializa la GRU del decodificador, que recibe la frase en inglés corrida un paso (*teacher forcing*).
Único cambio: `mask_zero=True` en el embedding del codificador, para que la dirección hacia atrás no empiece leyendo
relleno.""")
code("""ruta_s2s = RAIZ / "results" / f"seq2seq_gru{S2S_OCULTO}.keras"
s2s = None
if not ENTRENAR and ruta_s2s.exists():
    try:
        s2s = keras.models.load_model(ruta_s2s)
        print("Modelo cargado de", ruta_s2s.name)
    except Exception as e:
        print("No se pudo cargar el modelo guardado (", e, "); se reentrena.")
if s2s is None:
    s2s = construir_seq2seq(len(v_es), len(v_en), oculto=S2S_OCULTO)
    s2s.fit(Dtr[0], Dtr[1], sample_weight=Dtr[2], batch_size=64, epochs=15, validation_data=Dva,
            callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
    json.dump(s2s.history.history, open(RAIZ / "results" / "historia_traduccion.json", "w"))
    s2s.save(ruta_s2s)
s2s.summary(line_length=80)""")
code("""F.fig_curvas_traduccion()
display(Image(str(RAIZ / "figs" / "curvas_traduccion.png")))""")

md("""### 2.3 Traducción del conjunto de prueba
Decodificación voraz igual a `generate_translation` del notebook, pero en lotes: en el paso *i* se toma el argmax de la
posición *i* y se agrega a la entrada del decodificador hasta producir `[end]` o llegar a 20 palabras.""")
code("""hip = traducir_lote(s2s, v_es, v_en, te.es.tolist())
ref = te.en.tolist()
muestra = pd.DataFrame({"español": te.es.values, "referencia": ref, "predicción": hip}).sample(12, random_state=3)
muestra""")

md("""### 2.4 Métricas (implementación propia en `src/metricas.py`)
* **BLEU** de corpus: para n = 1..4 se cuentan los n-gramas de cada predicción, recortados al número de veces que aparecen
  en la referencia; p_n = coincidencias / total en todo el corpus. BLEU = BP · exp(¼ Σ log p_n), con penalización de
  brevedad BP = exp(1 − r/c) si la salida total es más corta que la referencia. Escala 0–100.
* **ROUGE-1 / ROUGE-2**: F de unigramas / bigramas compartidos. **ROUGE-L**: F con la subsecuencia común más larga
  (programación dinámica). Por frase y promediado. Escala 0–100.
* **F1** por palabras: precisión = palabras de la predicción que están en la referencia (contando repeticiones);
  recall = palabras de la referencia que aparecen en la predicción; F1 = media armónica. Por frase y promediado.

Las métricas se calculan sobre el texto normalizado (minúsculas, sin puntuación), que es la forma en que el modelo produce
el inglés. Como línea base se "traduce" copiando la frase en español sin cambios. Las implementaciones propias se validan
contra `sacrebleu` y `rouge-score` al final de esta sección.""")
code("""from evaluar_traduccion import calcular_y_guardar
t_test = te[["es", "en", "status"]].reset_index(drop=True).assign(prediccion=hip)
res, base, por_largo = calcular_y_guardar(t_test)   # guarda las tablas en results/
cols = ["BLEU", "ROUGE1", "ROUGE2", "ROUGEL", "F1", "Precision", "Recall", "exactas"]
tabla = pd.DataFrame([{k: res[k] for k in cols}, {k: base[k] for k in cols}],
                     index=["seq2seq GRU", "línea base: copiar el español"]).round(2)
print(res["BLEU_detalle"])
tabla""")
code("""F.fig_por_largo(); display(Image(str(RAIZ / "figs" / "metricas_por_largo.png")))
por_largo.round(2)""")
code("""F.fig_distribucion(); display(Image(str(RAIZ / "figs" / "distribucion_frases.png")))""")
md("""### 2.5 Validación de las métricas propias
Se comparan con las librerías de referencia sobre las mismas predicciones. BLEU propio usa palabras separadas por espacio
como tokens, lo que equivale a `sacrebleu` con `tokenize='none'`; el tokenizador por defecto de sacrebleu (`13a`) separa
algunos símbolos y da un valor ligeramente distinto, que se muestra como referencia.""")
code("""pd.read_csv(RAIZ / "results" / "validacion_metricas.csv").round(4)""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "accelerator": "GPU", "colab": {"provenance": []}}
salida = Path(__file__).parent / "Taller06_Generacion_y_Traduccion.ipynb"
nbf.write(nb, salida)
print("escrito", salida)
