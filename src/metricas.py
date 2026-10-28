"""Implementación propia de BLEU, ROUGE-N, ROUGE-L y F1 por palabras.

Se validan contra las librerías de referencia (sacrebleu y rouge-score) con
la función validar() de src/evaluar_traduccion.py; los resultados están en
results/validacion_metricas.csv.

Todas las funciones reciben texto ya normalizado (minúsculas, sin puntuación) y
devuelven valores en [0, 1].
"""
import math
import re
from collections import Counter


# ------------------------------------------------------------------ utilidades
def ngramas(tokens, n):
    """Bolsa (Counter) de n-gramas de una lista de tokens."""
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f(p, r):
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


# ------------------------------------------------------------------ BLEU
def bleu_corpus(hipotesis, referencias, n_max=4):
    """BLEU de corpus (Papineni et al., 2002), una referencia por frase.

    1. Para n = 1..4 se cuentan los n-gramas de cada predicción y se "recortan"
       (clipping) al número de veces que aparecen en su referencia; las cuentas
       se suman en todo el corpus: p_n = coincidencias_n / total_n.
    2. Penalización de brevedad: BP = 1 si c > r, si no exp(1 - r/c), con c y r
       las longitudes totales de predicciones y referencias.
    3. BLEU = BP * exp( (1/4) * sum(log p_n) ). Sin suavizado: si algún p_n es 0,
       BLEU es 0.
    Tokens = palabras separadas por espacio (equivale a sacrebleu tokenize='none').
    """
    coinc, total = [0] * n_max, [0] * n_max
    c = r = 0
    for h, ref in zip(hipotesis, referencias):
        th, tr = h.split(), ref.split()
        c += len(th); r += len(tr)
        for n in range(1, n_max + 1):
            gh, gr = ngramas(th, n), ngramas(tr, n)
            coinc[n - 1] += sum(min(k, gr[g]) for g, k in gh.items())
            total[n - 1] += max(len(th) - n + 1, 0)
    precisiones = [co / to if to else 0.0 for co, to in zip(coinc, total)]
    bp = 1.0 if c > r else (math.exp(1 - r / c) if c else 0.0)
    if min(precisiones) == 0:
        return 0.0, precisiones, bp
    return bp * math.exp(sum(math.log(p) for p in precisiones) / n_max), precisiones, bp


# ------------------------------------------------------------------ ROUGE
_NO_ALFANUM = re.compile(r"[^a-z0-9]+")


def tokens_rouge(texto):
    """Misma tokenización que rouge-score: minúsculas y solo [a-z0-9]."""
    return [t for t in _NO_ALFANUM.sub(" ", texto.lower()).split() if t]


def rouge_n(hip, ref, n):
    """ROUGE-N (Lin, 2004): F de n-gramas compartidos (con recorte)."""
    gh, gr = ngramas(tokens_rouge(hip), n), ngramas(tokens_rouge(ref), n)
    comunes = sum((gh & gr).values())
    th, tr = sum(gh.values()), sum(gr.values())
    if comunes == 0:
        return 0.0
    return _f(comunes / th, comunes / tr)


def lcs(a, b):
    """Longitud de la subsecuencia común más larga (programación dinámica)."""
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(hip, ref):
    """ROUGE-L: F con precisión = LCS/len(pred) y recall = LCS/len(ref)."""
    th, tr = tokens_rouge(hip), tokens_rouge(ref)
    if not th or not tr:
        return 0.0
    l = lcs(th, tr)
    return 0.0 if l == 0 else _f(l / len(th), l / len(tr))


# ------------------------------------------------------------------ F1
def f1_tokens(hip, ref):
    """F1 por palabras (bolsa con repeticiones, inspirado en SQuAD sin quitar artículos). Devuelve (P, R, F1)."""
    h, r = hip.split(), ref.split()
    if not h and not r:
        return 1.0, 1.0, 1.0
    comunes = sum((Counter(h) & Counter(r)).values())
    if comunes == 0:
        return 0.0, 0.0, 0.0
    p, rc = comunes / len(h), comunes / len(r)
    return p, rc, _f(p, rc)
