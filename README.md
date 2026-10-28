# Taller 06 — Generación de texto y traducción seq2seq (NLP, Javeriana)

**Autores:** David Moreno · Dierick Brochero
**Curso:** Procesamiento de Lenguaje Natural — Pontificia Universidad Javeriana

Adaptación del notebook de clase (*Deep Learning with Python*, 3a ed., cap. 15: "Language models and the Transformer")
a dos tareas:

1. **Generación de texto:** modelo de lenguaje por caracteres (GRU) entrenado con *Rebelión en la granja*
   (George Orwell, 1945; trad. de Marcial Souto).
2. **Traducción español → inglés:** seq2seq con GRU (codificador bidireccional) sobre *Combined Data es-ES*, evaluado con
   **BLEU**, **ROUGE-1/2/L** y **F1**, implementados en `src/metricas.py` y validados contra `sacrebleu` y `rouge-score`.

Reporte: [`Reporte_Taller06_Generacion_Traduccion.pdf`](Reporte_Taller06_Generacion_Traduccion.pdf) ·
Notebook: [`notebooks/Taller06_Generacion_y_Traduccion.ipynb`](notebooks/Taller06_Generacion_y_Traduccion.ipynb)

## Cómo se cumple cada punto del enunciado

| Punto del taller | Dónde está |
|---|---|
| 1. Usando el Colab del curso, entrenar con un libro leído por el grupo y generar texto | `notebooks/` (Parte 1), `src/libro.py`, `src/generacion.py`, `src/correr_generacion.py`; muestras en `results/muestras_generacion.json` |
| 2. Con *Combined Data es-ES*, entrenar un seq2seq español → inglés | `notebooks/` (Parte 2), `src/traduccion.py`, `src/correr_traduccion.py`; traducciones del test en `results/traducciones_test.csv` |
| 2. Implementar BLEU, ROUGE y F1 | `src/metricas.py` (implementación propia), `src/evaluar_traduccion.py`; validación en `results/validacion_metricas.csv` |
| 3. Reporte escrito | `Reporte_Taller06_Generacion_Traduccion.pdf` (fuente: `src/reporte.py`) |
| 4. Enlace a GitHub en el reporte | Portada y sección 4 del reporte |

## Resultados principales

| Tarea | Resultado |
|---|---|
| Generación (GRU 512, early stopping) | validación: 1,425 nats/carácter · **perplejidad 4,16** · exactitud 0,584 |
| Traducción es→en (GRU 1024), test | **BLEU 3,32** · ROUGE-1 **21,9** · ROUGE-2 5,7 · ROUGE-L **21,1** · F1 **21,8** |
| Línea base: copiar el español | BLEU 3,47 · ROUGE-1 10,2 · ROUGE-L 10,1 · F1 10,5 |

- **Generación:** con argmax (lo que hace el notebook de clase) el texto entra en ciclos ("de la casa de la casa…").
  El muestreo con temperatura da texto variado con el vocabulario y los personajes de la novela, pero sin coherencia.
  Con un libro tan corto, el modelo además copia frases literales del original (ver reporte, sección 1.5).
- **Traducción:** el modelo duplica a la línea base en ROUGE y F1, pero no la supera en BLEU: aprende el inglés frecuente,
  no a conservar el significado. Las causas (pocos pares, vocabulario grande, ningún mecanismo de atención) están analizadas
  en el reporte.

## Decisiones y por qué

| Decisión | Motivo |
|---|---|
| Dirección invertida (es → en) respecto al notebook | Lo pide el taller |
| Solo pares de ≤ 20 palabras (es) y ≤ 19 (en) | Un seq2seq sin atención resume toda la frase en un vector; la mediana del dataset es 53 / 62 palabras |
| Quitar 832 pares con "español" idéntico al inglés | No están traducidos (hashes, risas, otros idiomas); enseñarían a copiar e inflarían las métricas |
| `mask_zero=True` en el codificador | Sin máscara, la GRU bidireccional hacia atrás empieza leyendo relleno |
| 10 % final del libro como validación + *early stopping* | El notebook no valida el modelo de caracteres; así se mide en texto no visto y se para a tiempo |
| GRU de 512 (no 1024) en generación | Entrenamiento en CPU; con 1.540 ventanas la red de 512 ya sobreajusta. Se cambia con `GEN_OCULTO` en el notebook |
| Métricas propias además de las librerías | El taller pide implementarlas; se validan contra `sacrebleu` y `rouge-score` |
| Línea base de "copiar el español" | Para saber qué parte del puntaje es solapamiento trivial (nombres, URLs) |

## Estructura

```
data/
  Combined_Data_es-ES.csv.gz        dataset de traducción (incluido)
  Rebelion_en_la_granja.epub        NO incluido: ver "Sobre el libro"
src/
  libro.py              lectura del epub y limpieza (solo capítulos I–X)
  generacion.py         modelo de caracteres, generador paso a paso, muestreo con temperatura
  correr_generacion.py  entrena y guarda resultados de la parte 1
  evaluar_generacion.py palabras reales, repetición y copia literal según la temperatura
  traduccion.py         datos, vocabularios, seq2seq y decodificación voraz en lote
  metricas.py           BLEU, ROUGE-N, ROUGE-L y F1 (implementación propia)
  evaluar_traduccion.py métricas del test, por longitud y validación contra las librerías
  correr_traduccion.py  entrena, traduce el test y evalúa (parte 2)
  figuras.py            figuras del reporte
  reporte.py            genera el PDF; todas las cifras se leen de results/
notebooks/
  Taller06_Generacion_y_Traduccion.ipynb   notebook ejecutado, listo para Colab
  construir_notebook.py                    genera el notebook
results/   métricas, historias de entrenamiento, muestras, traducciones del test, modelo de caracteres
figs/      figuras del reporte
```

## Sobre el libro

*Rebelión en la granja* **no es de dominio público en Colombia**: Orwell murió en 1950 y el plazo colombiano es la vida
del autor más 80 años (hasta el 1 de enero de 2031). Se usa solo para entrenar el modelo en esta actividad académica sin
fines comerciales y **el texto no se distribuye en este repositorio** (`data/*.epub` está en `.gitignore`).
Para reproducir la parte 1 hay que poner una copia propia en `data/Rebelion_en_la_granja.epub`; en Colab, el notebook pide
subirla. El cargador toma del epub solo los documentos que empiezan con un número de capítulo (I–X).

## Cómo correrlo

**Colab:** abrir el notebook, elegir *Entorno de ejecución → Cambiar tipo → GPU T4* y ejecutar todo. El notebook clona este
repositorio, instala `sacrebleu` y `rouge-score` y pide subir el epub. El modelo de caracteres se carga de `results/`; el seq2seq
no está en el repositorio y se reentrena, así que sus métricas pueden variar un poco respecto al reporte. Con `ENTRENAR = True`
se reentrenan ambos. Las figuras y tablas de `results/` se regeneran con la corrida actual.

**Local:**
```bash
pip install -r requirements.txt
python src/correr_traduccion.py 1024 15   # ~35 min en CPU (2 núcleos)
python src/correr_generacion.py 512 60    # ~9 min en CPU
python src/evaluar_generacion.py          # calidad del texto según la temperatura
python src/evaluar_traduccion.py          # (opcional) recalcula las métricas desde las predicciones guardadas
python src/reporte.py                     # PDF
```

El modelo seq2seq entrenado (`results/seq2seq_gru1024.keras`, 351 MB) supera el límite de GitHub y no está en el
repositorio: el notebook lo reentrena si no lo encuentra. El modelo de caracteres (15 MB) sí está incluido.
Semilla 42; los resultados del reporte se obtuvieron con TensorFlow 2.21 y Keras 3.15 en CPU.
