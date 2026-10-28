"""Reporte escrito (PDF). Todas las cifras se leen de results/. Uso: python src/reporte.py"""
import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

sys.path.insert(0, str(Path(__file__).parent))
import figuras as F
from libro import PROMPTS

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "results"
SALIDA = RAIZ / "Reporte_Taller06_Generacion_Traduccion.pdf"
REPO = "https://github.com/davidm094/NLP_JAVE_TALLER06_202601"

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("F", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("FB", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("FI", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("FM", D + "DejaVuSansMono.ttf"))
pdfmetrics.registerFontFamily("F", normal="F", bold="FB", italic="FI", boldItalic="FB")

st = getSampleStyleSheet()
N = ParagraphStyle("N", parent=st["Normal"], fontName="F", fontSize=9.2, leading=12.8, spaceAfter=4)
H1 = ParagraphStyle("H1", parent=N, fontName="FB", fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=N, fontName="FB", fontSize=10.2, leading=13, spaceBefore=6, spaceAfter=3)
T = ParagraphStyle("T", parent=N, fontName="FB", fontSize=17, leading=21, spaceAfter=3)
S = ParagraphStyle("S", parent=N, fontSize=8, leading=10.5, textColor=colors.HexColor("#52514e"))
C = ParagraphStyle("C", parent=N, fontSize=7.8, leading=9.6, spaceAfter=0)
CB = ParagraphStyle("CB", parent=C, fontName="FB")
M = ParagraphStyle("M", parent=N, fontName="FM", fontSize=7.4, leading=9.4, leftIndent=8,
                   backColor=colors.HexColor("#f4f3ef"), borderPadding=4, spaceBefore=3, spaceAfter=6)
B = ParagraphStyle("B", parent=N, leftIndent=10, spaceAfter=2.5)
p = lambda t, s=N: Paragraph(t.replace('<code>', "<font name='FM' size='8'>").replace('</code>', '</font>'), s)
dec = lambda x, n=1: f"{x:.{n}f}".replace(".", ",")
mil = lambda x: f"{int(round(x)):,}".replace(",", ".")
bullet = lambda t: Paragraph(t.replace('<code>', "<font name='FM' size='8'>").replace('</code>', '</font>'), B, bulletText="•")
h1 = lambda t, alto=4: [CondPageBreak(alto * cm), p(t, H1)]


def tabla(filas, anchos, resaltar=()):
    data = [[p(str(c), CB if i == 0 else C) for c in fila] for i, fila in enumerate(filas)]
    t = Table(data, colWidths=[a * cm for a in anchos], repeatRows=1)
    est = [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe6")),
           ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9c8c2")),
           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2),
           ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    est += [("BACKGROUND", (0, r), (-1, r), colors.HexColor("#eef4fc")) for r in resaltar]
    t.setStyle(TableStyle(est))
    return t


def fig(nombre, ancho, pie):
    ruta = RAIZ / "figs" / nombre
    w, h = PILImage.open(ruta).size
    return KeepTogether([Image(str(ruta), width=ancho * cm, height=ancho * cm * h / w), p(pie, S)])


def bloque(texto, max_chars=None):
    t = texto if max_chars is None else texto[:max_chars]
    return p(escape(t).replace("\n", "<br/>"), M)


# ------------------------------------------------------------------ datos
F.todas()
G = json.load(open(RES / "resumen_generacion.json"))
MU = json.load(open(RES / "muestras_generacion.json"))
TR = json.load(open(RES / "resumen_traduccion.json"))
ML = pd.read_csv(RES / "metricas_por_largo.csv")
TT = pd.read_csv(RES / "traducciones_test.csv").fillna({"prediccion": ""})
HG = json.load(open(RES / "historia_generacion.json"))
PAR = json.load(open(RES / "parametros.json"))
m, base, dd = TR["metricas_test"], TR["linea_base_copiar_espanol"], TR["datos"]
muestra = lambda prompt, T: next(x["texto"] for x in MU if x["prompt"] == prompt and str(x["temperatura"]) == str(T))

E = []
E += [p("Taller 06 — Generación de texto y traducción seq2seq", T),
      p("Procesamiento de Lenguaje Natural · Pontificia Universidad Javeriana", S),
      p("David Moreno · Dierick Brochero", S),
      p(f'Código: <link href="{REPO}"><u>{REPO}</u></link>', S), Spacer(1, 6)]

E += h1("Resumen")
E += [p(
    f"Se adaptó el notebook de clase (<i>Deep Learning with Python</i>, 3a ed., cap. 15) a dos tareas. "
    f"(1) Un modelo de lenguaje por caracteres (GRU) entrenado con <i>Rebelión en la granja</i> de George Orwell "
    f"llega a una perplejidad de validación de <b>{dec(G['perplejidad_val'], 2)}</b> por carácter y genera texto con la "
    f"puntuación, los personajes y el tono de la novela, pero sin coherencia más allá de unas pocas palabras; con temperaturas "
    f"bajas, buena parte de lo que \"acierta\" es copia literal del libro. "
    f"(2) Un seq2seq GRU español→inglés entrenado con {mil(TR['train'])} pares cortos de <i>Combined Data es-ES</i> obtiene en test "
    f"<b>BLEU {dec(m['BLEU'], 2)}</b>, <b>ROUGE-1 {dec(m['ROUGE1'])}</b>, <b>ROUGE-L {dec(m['ROUGEL'])}</b> y <b>F1 {dec(m['F1'])}</b>. "
    f"Duplica a la línea base de copiar el español en ROUGE-1, ROUGE-L y F1, pero no la supera en BLEU: el modelo aprende vocabulario "
    f"frecuente del inglés, no a conservar el significado de la frase.")]

# ================================================================ PARTE 1
E += h1("1. Generación de texto con un libro")
E += [p("1.1 Libro y preparación", H2), p(
    f"<b>Libro:</b> <i>Rebelión en la granja</i> (<i>Animal Farm</i>, George Orwell, 1945), en la traducción al español de "
    f"Marcial Souto. Es un libro que leímos ambos integrantes del grupo. La obra <b>no es de dominio público en Colombia</b>: Orwell "
    f"murió en 1950 y aquí la protección dura la vida del autor más 80 años, hasta el 1 de enero de 2031. Se usa solo para entrenar "
    f"el modelo en esta actividad académica sin fines comerciales, y por eso el texto <b>no está en el repositorio</b>: el código lo "
    f"lee de una copia propia en epub (<code>data/Rebelion_en_la_granja.epub</code>, excluida en <code>.gitignore</code>).")]
E += [p(
    f"Del epub se toman solo los diez capítulos de la novela (I a X): quedan fuera la portada, los créditos, el prólogo de Orwell "
    f"(<i>La libertad de prensa</i>), el epílogo y las notas, que no son parte del relato. Se quitan las llamadas a notas y queda un "
    f"párrafo por línea. Resultado: <b>{mil(G['caracteres'])}</b> caracteres y un vocabulario de <b>{G['vocab']}</b> símbolos "
    f"(incluye relleno y [UNK]). Es un corpus pequeño: unas 6,5 veces menos texto que el de Shakespeare del notebook.")]
E += [p(
    f"Como en el notebook, se cortan ventanas no solapadas de 100 caracteres y la etiqueta es la misma ventana corrida un carácter. "
    f"<b>Cambio frente al notebook:</b> el último 10 % del libro queda como validación ({mil(G['secuencias_val'])} ventanas; "
    f"{mil(G['secuencias_train'])} de entrenamiento) para medir el modelo en texto no visto y detener el entrenamiento con "
    f"<i>early stopping</i> (paciencia 3, se restauran los mejores pesos).")]
E += [p("1.2 Modelo", H2), p(
    f"Embedding de 256 → GRU con <code>return_sequences</code> → Dropout 0,1 → Dense softmax sobre los {G['vocab']} símbolos; Adam y "
    f"entropía cruzada. El notebook usa una GRU de 1024 unidades; aquí se entrenó con <b>{G['oculto']}</b> para que corriera en CPU (no había GPU "
    f"disponible). No parece ser lo que limita al modelo: con solo {mil(G['secuencias_train'])} ventanas de entrenamiento, la red "
    f"de 512 ya sobreajusta (Figura 1), así que más unidades difícilmente ayudarían sin más texto. El notebook mantiene 512 para "
    f"reproducir este reporte; cambiar a 1024 es una línea (<code>GEN_OCULTO</code>).")]

ep_g, mej_g = G["epocas_corridas"], G["mejor_epoca"]
E += [p("1.3 Resultados del entrenamiento", H2),
      tabla([["Métrica", "Entrenamiento (mejor época)", "Validación (mejor época)"],
             ["Pérdida (nats por carácter)", dec(G["train_loss_mejor"], 3), dec(G["val_loss"], 3)],
             ["Perplejidad por carácter", dec(2.718281828 ** G["train_loss_mejor"], 2), dec(G["perplejidad_val"], 2)],
             ["Exactitud del siguiente carácter", dec(G["train_acc_mejor"], 3), dec(G["val_acc"], 3)]],
            [6, 5, 5]),
      Spacer(1, 4),
      p("La exactitud de entrenamiento es el promedio de la época con dropout activo; la de validación, sin dropout: "
        "no son estrictamente comparables.", S),
      p(f"El entrenamiento corrió {ep_g} épocas ({dec(G['minutos'])} min); la mejor validación fue en la época {mej_g}. "
        f"Una perplejidad de {dec(G['perplejidad_val'], 2)} significa que, en promedio, el modelo duda entre unos "
        f"{dec(G['perplejidad_val'], 1)} caracteres igual de probables en cada paso (de {G['vocab']} posibles)."),
      fig("curvas_generacion.png", 15.5, "Figura 1. Curvas de entrenamiento del modelo de caracteres. La línea punteada marca la época con menor pérdida de validación, cuyos pesos se conservan.")]

E += [p("1.4 Texto generado", H2), p(
    "El notebook genera con <b>argmax</b> (siempre el carácter más probable). Se comparó con <b>muestreo con temperatura</b> "
    "<i>T</i>: se divide el logaritmo de las probabilidades por <i>T</i> y se renormaliza. Muestras de 350 caracteres (semilla fija):")]
for prompt in PROMPTS[:2]:
    for Tm, nombre in [("voraz", "argmax (notebook)"), (0.5, "T = 0,5"), (1.0, "T = 1,0")]:
        E += [KeepTogether([p(f"<b>Prompt</b> «{escape(prompt)}» · {nombre}", C), bloque(muestra(prompt, Tm))])]
CG = pd.read_csv(RES / "calidad_generacion.csv")
cg = {r.estrategia: r for r in CG.itertuples()}
E += [Spacer(1, 3), p(
    "Para no juzgar solo a ojo, se generaron 500 caracteres por estrategia con los tres prompts (dos semillas cuando hay muestreo) "
    "y se midió: qué porcentaje de las palabras generadas existe en el libro, qué fracción de los bigramas de palabras es distinta "
    "(1 = no se repite nada), qué porcentaje de los 5-gramas de palabras generados aparece tal cual en el libro (copia literal) y "
    "el tramo más largo de palabras seguidas copiado del libro:"),
    tabla([["Estrategia", "Palabras generadas", "% palabras que existen en el libro", "Bigramas distintos / total",
            "% 5-gramas copiados del libro", "Tramo copiado más largo"]] +
          [[("argmax (notebook)" if r.estrategia == "argmax" else r.estrategia.replace("T=", "T = ").replace(".", ",")),
            r.palabras, dec(r.pct_palabras_reales), dec(r.bigramas_distintos, 2), dec(r.pct_5gramas_copiados),
            f"{r.tramo_copiado_max} palabras"] for r in CG.itertuples()],
          [3.0, 2.2, 3.0, 2.6, 2.8, 2.6]), Spacer(1, 4)]
E += [p("1.5 Análisis", H2),
      bullet(f"<b>Argmax entra en ciclos.</b> Siempre elige el carácter más probable y termina repitiendo «de la casa y los animales de la "
             f"casa…»: el {dec(cg['argmax'].pct_palabras_reales, 1)} % de las palabras son reales, pero solo el "
             f"{dec(cg['argmax'].bigramas_distintos * 100, 0)} % de los bigramas es distinto. Su {dec(cg['argmax'].pct_5gramas_copiados, 0)} % "
             "de 5-gramas «copiados» sale casi todo de dos 5-gramas (<i>de la casa y los</i>, <i>la casa y los animales</i>) que el ciclo "
             "repite entre 10 y 16 veces por prompt (81 de los 87 5-gramas copiados): no es memoria de pasajes. Es lo que daría el notebook de clase tal como está."),
      bullet(f"<b>Con T = 0,5 el texto se ve mejor en parte porque recita el libro.</b> El {dec(cg['T=0.5'].pct_5gramas_copiados, 1)} % de "
             f"sus 5-gramas está literal en la novela y llega a copiar {cg['T=0.5'].tramo_copiado_max} palabras seguidas. Un ejemplo de las muestras: «haber visto que "
             "te hablaba y que tú te dejabas acariciar» (11 palabras), que en el libro aparece una sola vez. Con un corpus tan corto, una distribución "
             "concentrada lleva al modelo a los caminos que vio en entrenamiento."),
      bullet(f"<b>Con T ≥ 0,8 deja de copiar pero inventa.</b> La copia literal baja a {dec(cg['T=0.8'].pct_5gramas_copiados, 1)} % "
             f"(T = 0,8) y {dec(cg['T=1.0'].pct_5gramas_copiados, 1)} % (T = 1,0), pero las palabras que existen bajan a "
             f"{dec(cg['T=0.8'].pct_palabras_reales, 0)} % y {dec(cg['T=1.0'].pct_palabras_reales, 0)} % (<i>camadadada</i>, <i>lugor</i>, "
             "<i>cezdíó</i>). No hay una temperatura que dé a la vez variedad, palabras reales y texto propio: es el límite de un modelo "
             "de caracteres entrenado con ~154 mil caracteres."),
      bullet("<b>Qué aprendió.</b> La forma del texto: el diálogo con raya («—¡Camaradas —dijo—, …»), las comillas latinas, la puntuación "
             "española (¡…!, ¿…?), concordancias cortas y el universo de la novela: <i>Napoleón</i>, <i>Bola de Nieve</i>, <i>Boxeador</i>, "
             "<i>Trébol</i>, <i>Jones</i>, <i>la Granja Animal</i>, <i>el molino de viento</i>."),
      bullet("<b>Qué no aprendió.</b> No hay coherencia más allá de unas pocas palabras y el texto no sigue al prompt: tras unas "
             "decenas de caracteres, las muestras de prompts distintos hablan de lo mismo (la casa, los animales, la granja). Por eso "
             "cada prompt usa su propia semilla: con la misma secuencia de números aleatorios, dos muestras pueden terminar idénticas."),
      bullet(f"<b>Sobreajuste.</b> Las curvas se separan desde la época ~12; la validación toca fondo en la época {mej_g} "
             f"({dec(G['val_loss'], 3)}) mientras el entrenamiento sigue bajando ({dec(HG['loss'][-1], 3)} en la época {ep_g}). "
             "Con ~154 mil caracteres de entrenamiento la red de 512 unidades tiene capacidad para memorizar; más texto (otras obras de "
             "Orwell) o una red más pequeña con más dropout serían el siguiente paso.")]

# ================================================================ PARTE 2
E += h1("2. Traducción español → inglés con seq2seq")
E += [p("2.1 Datos", H2), p(
    "<i>Combined Data es-ES</i> trae 53.043 textos de salud mental (Reddit/Twitter) con el original en inglés (<code>statement</code>) "
    "y su traducción al español (<code>statement_es_es</code>), así que sirve como corpus paralelo. La fuente es el español y el destino "
    "el inglés, con <code>[start]</code> y <code>[end]</code>, al revés del notebook (inglés→español).")]
E += [tabla([["Paso", "Pares que quedan", "Motivo"],
             ["Filas del archivo", mil(dd["filas"]), ""],
             ["Con texto en ambos idiomas", mil(dd["pares_validos"]), "se quitan vacíos"],
             ["≤ 20 palabras (es) y ≤ 19 (en)", mil(dd["cortos"]),
              "un seq2seq sin atención resume toda la frase en un vector; la mediana del dataset es de 53 palabras en español y 62 en inglés. 19 en inglés para que [start] + 19 + [end] quepan en las 21 posiciones del notebook"],
             ["Sin pares duplicados", mil(dd["sin_duplicados"]), ""],
             ["Sin pares idénticos es = en", mil(dd["sin_identicos"]),
              "no están traducidos (hashes, risas, otros idiomas, inglés sin traducir); enseñarían a copiar e inflarían las métricas"]],
            [4.3, 2.6, 9.3]),
      Spacer(1, 4),
      p(f"Normalización: minúsculas, sin puntuación ni ¿¡ y sin apóstrofo (<i>don't → dont</i>). Partición aleatoria 70/15/15 con semilla 42: "
        f"<b>{mil(TR['train'])}</b> de entrenamiento, <b>{mil(TR['val'])}</b> de validación y <b>{mil(TR['test'])}</b> de prueba. "
        f"Vocabularios (de entrenamiento, incluidos los tokens especiales): {mil(TR['vocab_es'])} en español y {mil(TR['vocab_en'])} en inglés; el "
        f"{dec(TR['unk_test_es'] * 100)} % de las palabras en español del test no aparece en entrenamiento.")]

E += [p("2.2 Modelo", H2), p(
    f"Arquitectura del notebook: embedding 256 → GRU bidireccional de {TR['oculto']} unidades (suma de direcciones) como "
    "codificador; su estado final inicializa la GRU del decodificador, que recibe la frase en inglés corrida un paso "
    "(<i>teacher forcing</i>) → Dropout 0,5 → Dense softmax. La pérdida ignora el relleno (<i>sample weights</i>). "
    "<b>Único cambio de arquitectura:</b> <code>mask_zero=True</code> en el embedding del codificador; sin máscara, la dirección "
    "hacia atrás de la GRU bidireccional empieza leyendo los ceros de relleno.")]
E += [p(
    f"Se entrenó con Adam, lotes de 64 y <i>early stopping</i> sobre la pérdida de validación (paciencia 3). Paró en la época "
    f"{TR['epocas_corridas']} y conservó la {TR['mejor_epoca']} (pérdida de validación {dec(TR['val_loss'], 3)}, exactitud por palabra "
    f"{dec(TR['val_acc'], 3)}); {dec(TR['minutos_entrenamiento'])} min en CPU. Desde la época {TR['mejor_epoca'] + 1} la pérdida de "
    "entrenamiento sigue bajando y la de validación sube: sobreajuste temprano por la poca cantidad de datos."),
    fig("curvas_traduccion.png", 15.5, "Figura 2. Curvas del seq2seq. La línea punteada marca la época conservada.")]

VM = pd.read_csv(RES / "validacion_metricas.csv")
vm = {r.metrica: r for r in VM.itertuples()}
E += [p("2.3 Métricas implementadas", H2), p(
    "Las tres métricas están implementadas desde cero en <code>src/metricas.py</code> y se validan contra las librerías estándar "
    "(<code>sacrebleu</code> y <code>rouge-score</code>)."),
      bullet("<b>BLEU</b> (Papineni et al., 2002), a nivel de corpus. Para n = 1…4 se cuentan los n-gramas de cada predicción y se "
             "recortan al número de veces que aparecen en su referencia (así repetir una palabra correcta no suma de más); "
             "p<sub>n</sub> = coincidencias / n-gramas predichos, sumando sobre todo el test. Penalización de brevedad: "
             "BP = exp(1 − r/c) si la longitud total de las predicciones (c) es menor que la de las referencias (r), si no 1. "
             "BLEU = BP · exp(¼ Σ log p<sub>n</sub>). Es exigente: sin 4-gramas correctos el puntaje cae aunque haya palabras sueltas bien."),
      bullet("<b>ROUGE-N</b> (Lin, 2004): con los n-gramas compartidos (recortados), precisión = compartidos / n-gramas de la "
             "predicción, recall = compartidos / n-gramas de la referencia y F = 2PR/(P+R). Se usa N = 1 y 2. "
             "<b>ROUGE-L</b>: igual, pero con la longitud de la subsecuencia común más larga (LCS, por programación dinámica). "
             "Por frase y promediadas."),
      bullet("<b>F1 por palabras</b> (inspirado en el F1 de SQuAD, sin quitar artículos): precisión = palabras de la predicción que están en la referencia (contando "
             "repeticiones); recall = palabras de la referencia que aparecen en la predicción; F1 = media armónica. Por frase y promediado. "
             "Es casi ROUGE-1 desde otro tokenizador: F1 usa las palabras tal cual (con ñ y tildes) y ROUGE solo a–z y 0–9, "
             "igual que rouge-score. Se interpretó «F1-score» como esta medida por palabras, la usual en generación de texto, y no como "
             "una combinación de BLEU y ROUGE."),
      p("Todas se calculan sobre texto normalizado (minúsculas, sin puntuación), que es la forma en que el modelo produce el inglés. "
        "La traducción del test se hace con decodificación voraz, el mismo algoritmo de <code>generate_translation</code> del notebook, "
        "pero en lotes. Como <b>línea base</b> se usa copiar la frase en español sin cambios."),
      KeepTogether([p("<b>Validación.</b> Sobre las 2.091 predicciones del test:", N),
      tabla([["Métrica", "Propia", "Librería", "Diferencia"],
             ["BLEU de corpus", dec(vm["BLEU (corpus)"].propia, 4), dec(vm["BLEU (corpus)"].libreria, 4) + " (sacrebleu, tokenize='none')",
              dec(abs(vm["BLEU (corpus)"].diferencia), 4)],
             ["ROUGE-1 promedio", dec(vm["ROUGE-1 (promedio)"].propia, 4), dec(vm["ROUGE-1 (promedio)"].libreria, 4) + " (rouge-score)",
              f"{dec(abs(vm['ROUGE-1 (promedio)'].diferencia), 4)}; máx. por frase {dec(vm['ROUGE-1 (promedio)'].max_dif_por_frase, 4)}"],
             ["ROUGE-2 promedio", dec(vm["ROUGE-2 (promedio)"].propia, 4), dec(vm["ROUGE-2 (promedio)"].libreria, 4) + " (rouge-score)",
              f"{dec(abs(vm['ROUGE-2 (promedio)'].diferencia), 4)}; máx. por frase {dec(vm['ROUGE-2 (promedio)'].max_dif_por_frase, 4)}"],
             ["ROUGE-L promedio", dec(vm["ROUGE-L (promedio)"].propia, 4), dec(vm["ROUGE-L (promedio)"].libreria, 4) + " (rouge-score)",
              f"{dec(abs(vm['ROUGE-L (promedio)'].diferencia), 4)}; máx. por frase {dec(vm['ROUGE-L (promedio)'].max_dif_por_frase, 4)}"]],
            [3.4, 2.0, 5.6, 5.2])]),
      Spacer(1, 3),
      p(f"El BLEU propio toma como tokens las palabras separadas por espacio. El tokenizador por defecto de sacrebleu (13a) separa "
        f"además algunos símbolos que quedaron en el texto y da {dec(vm['BLEU (corpus), tokenizador 13a'].libreria, 2)}; la diferencia "
        f"no cambia ninguna conclusión.", S)]

fila = lambda nombre, r: [nombre, dec(r["BLEU"], 2), dec(r["ROUGE1"]), dec(r["ROUGE2"]), dec(r["ROUGEL"]),
                          dec(r["F1"]), dec(r["Precision"]), dec(r["Recall"])]
E += [p("2.4 Resultados en test", H2),
      tabla([["Sistema", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "F1", "Precisión", "Recall"],
             fila("seq2seq GRU", m), fila("Línea base: copiar el español", base)],
            [4.4, 1.3, 1.6, 1.6, 1.6, 1.3, 2.0, 1.6], resaltar=(1,)),
      Spacer(1, 3),
      p(f"Detalle de BLEU del modelo: <font name='FM' size='7.5'>{escape(m['BLEU_detalle'])}</font>. "
        f"De la línea base: <font name='FM' size='7.5'>{escape(base['BLEU_detalle'])}</font>. Escala 0–100 en todas.", S)]

filas_l = [["Longitud (palabras en español)", "Frases", "BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "F1"]]
filas_l += [[r.tramo, r.n, dec(r.BLEU, 2), dec(r.ROUGE1), dec(r.ROUGE2), dec(r.ROUGEL), dec(r.F1)] for r in ML.itertuples()]
E += [Spacer(1, 4), tabla(filas_l, [4.6, 1.5, 1.6, 1.7, 1.7, 1.7, 1.6]), Spacer(1, 4),
      fig("metricas_por_largo.png", 15.5, "Figura 3. Métricas en test por tramo de longitud de la frase en español."),
      fig("distribucion_frases.png", 15.5, "Figura 4. Distribución de F1 y BLEU por frase en el test. El BLEU por frase de esta figura lo calcula "
          "sacrebleu.sentence_bleu con suavizado (sin suavizado casi todas las frases darían 0); las cifras de BLEU de las tablas son de corpus, con la implementación propia.")]

EJ = [476, 17, 1203, 834, 54, 1659, 81, 1055, 756]  # filas de results/traducciones_test.csv
filas_e = [["Español (entrada)", "Referencia (inglés)", "Predicción del modelo", "F1"]]
filas_e += [[escape(TT.es[i]), escape(TT.en[i]), escape(TT.prediccion[i]), dec(TT.f1_tok[i] * 100, 0)] for i in EJ]
f1q = TT.f1_tok.quantile([0.25, 0.5, 0.75]) * 100
frec = TT.prediccion[TT.prediccion != ''].value_counts()
n_vacias = int((TT.prediccion == '').sum())
E += [p("2.5 Ejemplos del test", H2), p(
    "Ejemplos elegidos para cubrir aciertos, aciertos parciales y fallos (F1 por frase, 0–100). Se evitaron textos con contenido "
    "sensible del dataset; todas las predicciones están en <code>results/traducciones_test.csv</code>."),
      tabla(filas_e, [4.6, 4.6, 4.6, 1.0]), Spacer(1, 4)]

E += [p("2.6 Análisis", H2),
      bullet(f"<b>El modelo aprende el inglés frecuente, no la traducción.</b> La mediana de F1 por frase es {dec(f1q[0.5], 0)} "
             f"(cuartiles {dec(f1q[0.25], 0)}–{dec(f1q[0.75], 0)}) y el {dec((TT.f1_tok == 0).mean() * 100, 0)} % de las frases no "
             "comparte ninguna palabra con la referencia. Las frases cortas con estructuras comunes (<i>i dont like…</i>, "
             "<i>i have to go…</i>) salen bien; en el resto el decodificador arranca bien y luego sigue con una continuación "
             "frecuente en inglés (<i>bathroom → bank</i>, <i>job → life</i>)."),
      bullet(f"<b>Colapso a frases genéricas.</b> Hay predicciones que se repiten para entradas distintas: «{escape(frec.index[0])}» aparece "
             f"{frec.iloc[0]} veces en el test, y {n_vacias} frases se tradujeron como vacías: el primer token predicho fue [end], casi siempre con entradas de una o dos palabras (<i>mañana</i>, <i>cemento</i>, <i>nombre</i>). Una posible explicación: con decodificación voraz, cuando la entrada aporta poca evidencia, el decodificador "
             "sigue la secuencia más probable según su modelo de lenguaje en inglés; en entrenamiento hay 20 frases con "
             "<i>so nervous</i> (todas de la clase Anxiety). Es una hipótesis: no se verificó con otra estrategia de decodificación."),
      bullet(f"<b>BLEU frente a ROUGE/F1.</b> En ROUGE-1, ROUGE-L y F1 el modelo duplica a la línea base de copiar el español "
             f"(ROUGE-1 {dec(m['ROUGE1'])} frente a {dec(base['ROUGE1'])}; F1 {dec(m['F1'])} frente a {dec(base['F1'])}; en ROUGE-2 la "
             f"ventaja es menor: {dec(m['ROUGE2'])} frente a {dec(base['ROUGE2'])}), pero en BLEU queda prácticamente igual o por debajo "
             f"({dec(m['BLEU'], 2)} frente a {dec(base['BLEU'], 2)}). La copia acierta 4-gramas completos con nombres de usuario, "
             "URLs y nombres propios que el traductor del dataset dejó igual (los 4-gramas que acierta la copia son cosas como "
             "<i>http plurk com p</i> o cadenas de usuarios de Twitter), y el modelo los pierde porque están fuera de su vocabulario. BLEU castiga mucho no tener 4-gramas correctos; por eso para un sistema tan débil ROUGE-1 y F1 "
             "distinguen mejor que BLEU."),
      bullet(f"<b>La longitud importa.</b> BLEU baja de {dec(ML.BLEU.iloc[0], 1)} en frases de 1–5 palabras a "
             f"{dec(ML.BLEU.iloc[-1], 1)} en 16–20; F1 va de {dec(ML.F1.iloc[0])} a {dec(ML.F1.iloc[-1])} (ROUGE-1 y F1 "
             "son casi iguales en 1–5 y 6–10 palabras y caen desde 11). Es consistente con el cuello de botella de un seq2seq sin "
             "atención, donde toda la frase pasa por un único vector de estado, aunque aquí no se probó un modelo con atención para confirmarlo."),
      bullet(f"<b>Sobreajuste temprano.</b> Con {mil(TR['train'])} pares y vocabularios de {mil(TR['vocab_es'])} y "
             f"{mil(TR['vocab_en'])} palabras, el 77 % de las palabras distintas aparece una o dos veces en entrenamiento (en ambos "
             f"idiomas); el modelo, de {dec(PAR['seq2seq'] / 1e6)} millones de parámetros, memoriza desde la época "
             f"{TR['mejor_epoca'] + 1}."),
      bullet("<b>Calidad del dataset.</b> La referencia (inglés) es el texto original; lo traducido es la entrada en español. Además de "
             "los pares sin traducir que se quitaron, en los textos largos del archivo el español tiene bastantes menos palabras que el "
             "inglés, lo que sugiere recortes en la traducción; en las frases cortas que se usaron la proporción es cercana a 1, así que "
             "el efecto aquí es menor (mediana es/en = 1,0). La entrada es lenguaje informal de redes (errores, jerga, nombres de usuario), "
             "a diferencia de las frases de manual del corpus spa-eng del notebook."),
      bullet("<b>Solapamiento entre particiones.</b> La deduplicación es por par (español, inglés): 27 frases en español del test "
             "(1,3 %) también están en entrenamiento con otra traducción, y 37 referencias en inglés (1,8 %). Es poco y no cambia las "
             "conclusiones, pero una versión más estricta deduplicaría por la frase en español antes de particionar.")]

E += h1("3. Conclusiones y limitaciones")
E += [bullet(f"<b>Generación:</b> con un GRU de {G['oculto']} unidades y un solo libro se obtiene un modelo que imita la superficie del "
             f"texto (perplejidad {dec(G['perplejidad_val'], 2)} por carácter en validación), pero no su sentido. La estrategia de "
             "decodificación importa tanto como el modelo: argmax, la del notebook, produce ciclos; T = 0,5 da texto legible pero en "
             f"parte copiado del libro ({dec(cg['T=0.5'].pct_5gramas_copiados, 1)} % de 5-gramas literales); T ≥ 0,8 es texto propio pero "
             "con muchas palabras inventadas. Con un libro de ~170 mil caracteres no hay un punto que lo resuelva todo."),
      bullet(f"<b>Traducción:</b> el seq2seq GRU sin atención, con ~10 mil pares cortos y ruidosos, logra BLEU {dec(m['BLEU'], 2)}, "
             f"ROUGE-L {dec(m['ROUGEL'])} y F1 {dec(m['F1'])}. Es un resultado bajo: traduce bien solo frases cortas y frecuentes. "
             "Si solo se hubiera reportado BLEU, parecería que no aprendió nada (no le gana a copiar el español); si solo ROUGE-1, que es "
             "el doble de bueno que la copia. Las dos lecturas son parciales."),
      bullet("<b>Limitaciones:</b> entrenamiento en CPU (GRU de 512 en lugar de 1024 en el modelo de caracteres), una sola semilla por "
             "modelo (no se estima la variabilidad entre corridas), decodificación voraz, métricas sobre texto en minúsculas y sin puntuación, "
             "una entrada en español traducida (no escrita por hablantes nativos) y un pequeño solapamiento de frases entre particiones."),
      bullet(f"<b>Siguientes pasos, cada uno por un problema medido:</b> atención o el Transformer de la siguiente sección del capítulo, "
             f"por la caída con la longitud (BLEU {dec(ML.BLEU.iloc[0], 1)} → {dec(ML.BLEU.iloc[-1], 1)}); búsqueda en haz, por las "
             f"{n_vacias} salidas vacías y las frases genéricas repetidas; subpalabras (BPE), porque el {dec(TR['unk_test_es'] * 100)} % "
             "de las palabras del test no está en el vocabulario y el 77 % del vocabulario aparece una o dos veces; y en generación, "
             "más texto del mismo autor, porque el modelo copia frases enteras con T = 0,5.")]

E += h1("4. Reproducibilidad", 3)
E += [p(f'Repositorio: <link href="{REPO}"><u>{REPO}</u></link>. '
        "<code>notebooks/Taller06_Generacion_y_Traduccion.ipynb</code> está listo para Colab (clona el repositorio e instala "
        "<code>sacrebleu</code> y <code>rouge-score</code>); con <code>ENTRENAR = True</code> reentrena ambos modelos. Los resultados de "
        "este reporte salen de <code>python src/correr_traduccion.py 1024 15</code> y <code>python src/correr_generacion.py 512 60</code> (más <code>python src/evaluar_generacion.py</code> para la tabla de calidad del texto; "
        "requiere la copia propia del libro en <code>data/</code>) "
        "(semilla 42; TensorFlow 2.21, Keras 3.15, CPU), guardados en <code>results/</code>. El PDF se regenera con "
        "<code>python src/reporte.py</code>.")]

if __name__ == "__main__":
    doc = SimpleDocTemplate(str(SALIDA), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Taller 06 — Generación y traducción",
                            author="David Moreno, Dierick Brochero")
    doc.build(E)
    print("escrito", SALIDA)
