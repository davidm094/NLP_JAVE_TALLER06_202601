"""Diagramas de la solución: pipelines de PLN (mismo estilo que el Taller 01),
arquitectura de los dos modelos y arquitectura del código.

Las cifras (vocabularios, tamaños, parámetros) se leen de results/.
Uso: python src/diagramas.py  ->  figs/pipeline_*.png, figs/arquitectura_*.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

RAIZ = Path(__file__).resolve().parents[1]
RES, FIGS = RAIZ / "results", RAIZ / "figs"
FIGS.mkdir(exist_ok=True)
OSCURO, CLARO, GRIS, TINTA = "#14375A", "#2E9BD6", "#8A8A85", "#1E2A2F"
NARANJA = "#D9822B"
mil = lambda x: f"{int(x):,}".replace(",", ".")


# ------------------------------------------------------------------ pipelines
def _chevron(ax, x, y, w, h, color, punta=0.28, primero=False):
    pts = [(x, y), (x + w - punta, y), (x + w, y + h / 2), (x + w - punta, y + h), (x, y + h)]
    if not primero:
        pts.append((x + punta, y + h / 2))
    ax.add_patch(Polygon(pts, closed=True, color=color, lw=0))


def pipeline(etapas, titulo, archivo, por_fila=4):
    """etapas: lista de (nombre, detalle). Chevrons numerados, colores alternos, en filas de `por_fila`."""
    n = len(etapas)
    filas = (n + por_fila - 1) // por_fila
    w, h, sep, sep_y = 2.2, 1.55, 0.12, 0.35
    ancho = por_fila * (w + sep)
    alto = filas * h + (filas - 1) * sep_y
    fig, ax = plt.subplots(figsize=(por_fila * 2.3, 0.75 + filas * 1.75), dpi=170)
    for i, (nombre, detalle) in enumerate(etapas):
        f, c = divmod(i, por_fila)
        x = c * (w + sep)
        y = alto - (f + 1) * h - f * sep_y
        _chevron(ax, x, y, w, h, OSCURO if i % 2 == 0 else CLARO, primero=(c == 0))
        cx = x + w / 2 + (0.06 if c else -0.06)
        ax.text(cx, y + h * 0.70, f"{i}\n{nombre}", ha="center", va="center", color="white",
                fontsize=11, fontweight="bold", linespacing=1.15)
        ax.text(cx, y + h * 0.28, detalle, ha="center", va="center", color="white", fontsize=8.8, linespacing=1.2)
    ax.set_xlim(-0.1, ancho)
    ax.set_ylim(-0.1, alto + 0.55)
    ax.text(ancho / 2, alto + 0.33, titulo, ha="center", va="center", fontsize=13, fontweight="bold", color=TINTA)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGS / archivo, facecolor="white")
    plt.close(fig)


def pipelines():
    G = json.load(open(RES / "resumen_generacion.json"))
    T = json.load(open(RES / "resumen_traduccion.json"))
    d = T["datos"]
    pipeline([
        ("Carga", "epub propio\ncapítulos I–X"),
        ("Limpieza", "sin notas ni\nprólogo; 1 párrafo\npor línea"),
        ("Tokenización", f"por carácter\n{G['vocab']} símbolos\n(vocab. de train)"),
        ("Ventanas", f"100 caracteres\nx → y corrido 1\n90 % / 10 %"),
        ("Modelo", f"Embedding 256\nGRU {G['oculto']}\nDense softmax"),
        ("Entrenamiento", f"Adam, lotes 64\nearly stopping\nmejor época {G['mejor_epoca']}"),
        ("Generación", "argmax y\ntemperatura\nT = 0,5 · 0,8 · 1,0"),
        ("Evaluación", f"perplejidad {G['perplejidad_val']:.2f}\npalabras reales\ncopia literal".replace(".", ",")),
    ], "Parte 1 · Pipeline de generación de texto (Rebelión en la granja)", "pipeline_generacion.png")
    pipeline([
        ("Carga", f"Combined Data\nes-ES\n{mil(d['filas'])} filas"),
        ("Limpieza", "minúsculas, sin\npuntuación ni\napóstrofos"),
        ("Filtrado", f"≤ 20 palabras\nsin duplicados\nni es = en: {mil(d['sin_identicos'])}"),
        ("Partición", f"70 / 15 / 15\ntrain {mil(T['train'])}\nval {mil(T['val'])} · test {mil(T['test'])}"),
        ("Tokenización", f"por palabra\nvocab. de train\n{mil(T['vocab_es'])} es · {mil(T['vocab_en'])} en"),
        ("Seq2seq", f"BiGRU {T['oculto']} →\nGRU {T['oculto']}\nteacher forcing"),
        ("Decodificación", "voraz (argmax)\nen lotes\nhasta [end]"),
        ("Evaluación", "BLEU · ROUGE · F1\nimplem. propia\n+ línea base"),
    ], "Parte 2 · Pipeline de traducción español → inglés", "pipeline_traduccion.png")


# ------------------------------------------------------------------ modelos
def _capa(ax, x, y, w, h, titulo, detalle, color, texto="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h * 0.64, titulo, ha="center", va="center", fontsize=9.5, fontweight="bold", color=texto)
    ax.text(x + w / 2, y + h * 0.28, detalle, ha="center", va="center", fontsize=7.8, color=texto)


def _flecha(ax, x1, y1, x2, y2, texto=None, lado="right"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GRIS, lw=1.3, shrinkA=0, shrinkB=0))
    if texto:
        ax.text((x1 + x2) / 2 + (0.08 if lado == "right" else -0.08), (y1 + y2) / 2, texto, fontsize=7.4,
                color="#4A5568", ha="left" if lado == "right" else "right", va="center", family="monospace")


def arquitectura_modelos():
    G = json.load(open(RES / "resumen_generacion.json"))
    T = json.load(open(RES / "resumen_traduccion.json"))
    P = json.load(open(RES / "parametros.json"))
    V, H = G["vocab"], G["oculto"]
    ve, vn, h2 = T["vocab_es"], T["vocab_en"], T["oculto"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 6.2), dpi=150, gridspec_kw={"width_ratios": [1, 1.55]})
    for ax in (a, b):
        ax.set_xlim(0, 10); ax.set_ylim(-0.3, 11); ax.axis("off")

    # ---- generación (pila vertical)
    a.set_title(f"Parte 1 · Modelo de lenguaje por caracteres\n{mil(P['generacion'])} parámetros",
                fontsize=11, fontweight="bold", color=TINTA)
    x, w, hh = 2.2, 5.6, 1.15
    pila = [("Entrada", "100 ids de caracteres", "#E6E2D8", TINTA),
            ("Embedding", f"{V} símbolos → 256", CLARO, "white"),
            ("GRU", f"{H} unidades, return_sequences", OSCURO, "white"),
            ("Dropout 0,1", "solo en entrenamiento", "#C9D3D6", TINTA),
            ("Dense + softmax", f"{V} clases por posición", CLARO, "white"),
            ("Salida", "siguiente carácter en cada posición", "#E6E2D8", TINTA)]
    formas = ["(lote, 100)", "(lote, 100, 256)", f"(lote, 100, {H})", f"(lote, 100, {H})", f"(lote, 100, {V})"]
    ys = [9.4 - i * 1.6 for i in range(len(pila))]
    for (t, dsc, c, tc), y in zip(pila, ys):
        _capa(a, x, y, w, hh, t, dsc, c, tc)
    for i in range(len(pila) - 1):
        _flecha(a, x + w / 2, ys[i], x + w / 2, ys[i + 1] + hh, formas[i])
    a.text(5, 0.25, "Generación: el mismo modelo paso a paso, con el estado de la GRU explícito;\n"
                    "el siguiente carácter sale por argmax o por muestreo con temperatura.",
           ha="center", fontsize=7.8, color="#4A5568")

    # ---- seq2seq (codificador | decodificador)
    b.set_title(f"Parte 2 · Seq2seq español → inglés\n{mil(P['seq2seq'])} parámetros",
                fontsize=11, fontweight="bold", color=TINTA)
    xe, xd, w2 = 0.0, 5.8, 4.2
    enc = [("Entrada (español)", "≤ 20 ids de palabras", "#E6E2D8", TINTA),
           ("Embedding", f"{mil(ve)} → 256, mask_zero", CLARO, "white"),
           ("GRU bidireccional", f"{h2} unidades, suma de direcciones", OSCURO, "white")]
    dec = [("Entrada (inglés)", "[start] + palabras previas", "#E6E2D8", TINTA),
           ("Embedding", f"{mil(vn)} → 256", CLARO, "white"),
           ("GRU", f"{h2} unidades, return_sequences", OSCURO, "white"),
           ("Dropout 0,5", "solo en entrenamiento", "#C9D3D6", TINTA),
           ("Dense + softmax", f"{mil(vn)} palabras por posición", CLARO, "white"),
           ("Salida", "siguiente palabra en inglés", "#E6E2D8", TINTA)]
    ye = [9.4, 7.8, 6.2]
    yd = [9.4 - i * 1.6 for i in range(len(dec))]
    for (t, dsc, c, tc), y in zip(enc, ye):
        _capa(b, xe, y, w2, hh, t, dsc, c, tc)
    for (t, dsc, c, tc), y in zip(dec, yd):
        _capa(b, xd, y, w2, hh, t, dsc, c, tc)
    for i in range(len(enc) - 1):
        _flecha(b, xe + w2 / 2, ye[i], xe + w2 / 2, ye[i + 1] + hh)
    for i in range(len(dec) - 1):
        _flecha(b, xd + w2 / 2, yd[i], xd + w2 / 2, yd[i + 1] + hh)
    # estado del codificador -> estado inicial de la GRU del decodificador
    b.annotate("", xy=(xd, yd[2] + hh / 2), xytext=(xe + w2, ye[2] + hh / 2),
               arrowprops=dict(arrowstyle="-|>", color=NARANJA, lw=2.2))
    b.text((xe + w2 + xd) / 2, ye[2] + hh / 2 + 0.55, f"estado\nfinal\n(lote, {h2})", ha="center", fontsize=7.4,
           color=NARANJA, fontweight="bold")
    b.text(xe + w2 / 2, 4.6, "Codificador", ha="center", fontsize=10.5, fontweight="bold", color=OSCURO)
    b.text(xe + w2 / 2, 4.1, "Lee toda la frase en español y la resume\nen un solo vector de estado.\n"
                             "Sin atención: el decodificador no vuelve\na mirar las palabras de entrada.",
           ha="center", fontsize=7.8, color="#4A5568", va="top")
    b.text(5, 0.25, "Entrenamiento con teacher forcing (el decodificador recibe la frase correcta corrida un paso); "
                    "en prueba,\ndecodificación voraz: cada palabra predicha se agrega a la entrada hasta producir [end].",
           ha="center", fontsize=7.8, color="#4A5568")
    fig.tight_layout()
    fig.savefig(FIGS / "arquitectura_modelos.png", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ código
def arquitectura_codigo():
    fig, ax = plt.subplots(figsize=(12, 4.6), dpi=150)
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    col = lambda x, titulo: ax.text(x, 4.35, titulo, ha="center", fontsize=10.5, fontweight="bold", color=TINTA)
    col(1.25, "Datos"); col(4.55, "Código (src/)"); col(7.95, "Resultados (results/)"); col(10.75, "Entregables")

    def caja(x, y, t, d, c, w=2.3, h=0.78, tc="white"):
        _capa(ax, x - w / 2, y, w, h, t, d, c, tc)

    caja(1.25, 2.9, "Rebelión en la granja", "epub propio (no se sube)", "#E6E2D8", tc=TINTA)
    caja(1.25, 1.2, "Combined Data es-ES", "csv.gz, 53.043 filas", "#E6E2D8", tc=TINTA)
    caja(4.55, 3.25, "libro.py · generacion.py", "carga, modelo, muestreo", OSCURO, w=2.9)
    caja(4.55, 2.35, "correr_ / evaluar_generacion", "entrena · mide calidad", CLARO, w=2.9)
    caja(4.55, 1.45, "traduccion.py · metricas.py", "seq2seq · BLEU/ROUGE/F1", OSCURO, w=2.9)
    caja(4.55, 0.55, "correr_ / evaluar_traduccion", "entrena · traduce · valida", CLARO, w=2.9)
    caja(7.95, 2.75, "Parte 1", "modelo .keras, historia,\nmuestras, calidad", "#C9D3D6", w=2.5, h=1.0, tc=TINTA)
    caja(7.95, 0.85, "Parte 2", "historia, traducciones,\nmétricas, validación", "#C9D3D6", w=2.5, h=1.0, tc=TINTA)
    caja(10.75, 3.1, "figuras.py · diagramas.py", "figs/*.png", NARANJA, w=2.4)
    caja(10.75, 1.9, "reporte.py", "Reporte PDF", NARANJA, w=2.4)
    caja(10.75, 0.7, "construir_notebook.py", "notebook (Colab)", NARANJA, w=2.4)
    for (x1, y1, x2, y2) in [(2.4, 3.29, 3.1, 3.6), (2.4, 3.29, 3.1, 2.75), (2.4, 1.59, 3.1, 1.85), (2.4, 1.59, 3.1, 0.95),
                             (6.0, 3.0, 6.7, 3.25), (6.0, 1.0, 6.7, 1.35),
                             (9.2, 3.25, 9.55, 3.5), (9.2, 3.25, 9.55, 2.3), (9.2, 1.35, 9.55, 2.3), (9.2, 1.35, 9.55, 1.1)]:
        _flecha(ax, x1, y1, x2, y2)
    ax.text(6, 0.02, "Todas las cifras del reporte se leen de results/: si se reentrena, el PDF y las figuras se regeneran sin editar texto a mano.",
            ha="center", fontsize=8, color="#4A5568")
    fig.tight_layout()
    fig.savefig(FIGS / "arquitectura_codigo.png", facecolor="white")
    plt.close(fig)


def todas():
    pipelines(); arquitectura_modelos(); arquitectura_codigo()


if __name__ == "__main__":
    todas()
    print("diagramas en", FIGS)
