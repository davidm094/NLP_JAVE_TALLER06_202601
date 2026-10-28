"""Carga y limpieza del libro para el modelo de lenguaje por caracteres.

Libro: "Rebelión en la granja" (Animal Farm), George Orwell, 1945;
traducción al español de Marcial Souto.

La obra NO es de dominio público en Colombia (Orwell murió en 1950; el plazo
colombiano es la vida del autor + 80 años, hasta el 1 de enero de 2031), así
que el texto no se distribuye en este repositorio: cada usuario debe poner su
propia copia en formato epub en data/ (ver ARCHIVO). Se usa solo para entrenar
el modelo en una actividad académica sin fines comerciales.
"""
import html
import re
import zipfile
from pathlib import Path, PurePosixPath

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "data" / "Rebelion_en_la_granja.epub"
TITULO = "Rebelión en la granja"
AUTOR = "George Orwell (1945), trad. de Marcial Souto"
# Prompts para la generación: personajes y un arranque típico de diálogo de la novela
PROMPTS = ["Napoleón", "—¡Camaradas", "Bola de Nieve"]


def _spine(z):
    """Archivos xhtml del epub en el orden de lectura (spine del content.opf)."""
    contenedor = z.read("META-INF/container.xml").decode("utf-8")
    opf = re.search(r'full-path="([^"]+)"', contenedor).group(1)
    base = PurePosixPath(opf).parent
    s = z.read(opf).decode("utf-8")
    items = dict(re.findall(r'<item\b[^>]*\bid="([^"]+)"[^>]*\bhref="([^"]+)"', s))
    items.update({i: h for h, i in re.findall(r'<item\b[^>]*\bhref="([^"]+)"[^>]*\bid="([^"]+)"', s)})
    return [str(base / items[i]) for i in re.findall(r'<itemref\b[^>]*\bidref="([^"]+)"', s)]


def _xhtml_a_parrafos(x):
    """Texto plano: un elemento de bloque (p, h1..h6, div) por línea."""
    x = re.sub(r"(?s)<head.*?</head>", "", x)
    x = re.sub(r"<(?:br\s*/?|/p|/h\d|/div)>", "\n", x)
    x = re.sub(r"<[^>]+>", "", x)
    x = html.unescape(x)
    return [" ".join(l.split()) for l in x.split("\n") if l.strip()]


def cargar_capitulos(ruta=ARCHIVO):
    """Lista de capítulos (cada uno, lista de párrafos) del cuerpo de la novela.

    Del epub se toman solo los documentos cuyo primer párrafo es un número romano
    (I ... X): así quedan fuera la portada, la sinopsis, los créditos, el prólogo de
    Orwell ("La libertad de prensa"), el epílogo y las notas, que no son la novela.
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(
            f"No está {ruta}. El libro no se incluye en el repositorio por derechos de autor: "
            "copie su propia copia en epub con ese nombre (en Colab, súbala con la celda de carga).")
    caps = []
    with zipfile.ZipFile(ruta) as z:
        for doc in _spine(z):
            par = _xhtml_a_parrafos(z.read(doc).decode("utf-8"))
            if par and re.fullmatch(r"[IVXL]+", par[0]):
                caps.append(par)
    return caps


def limpiar(capitulos):
    """Un párrafo por línea; el número del capítulo queda como línea propia.
    Se quitan las llamadas a notas '[*]' y se normalizan espacios."""
    lineas = []
    for cap in capitulos:
        for p in cap:
            p = p.replace("[*]", "").strip()
            if p:
                lineas.append(p)
    return "\n".join(lineas) + "\n"


def cargar_libro(ruta=ARCHIVO):
    return limpiar(cargar_capitulos(ruta))


if __name__ == "__main__":
    caps = cargar_capitulos()
    t = limpiar(caps)
    print(len(caps), "capítulos;", len(t), "caracteres;", len(set(t)), "caracteres distintos")
