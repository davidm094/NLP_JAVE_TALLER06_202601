"""Punto 2: traducción español -> inglés con seq2seq (GRU) y métricas BLEU, ROUGE y F1.

Sigue la sección "Sequence-to-sequence learning with RNNs" del notebook del curso,
con estos cambios:
  - dirección invertida: fuente = español (`statement_es_es`), destino = inglés
    (`statement`), con los marcadores [start] / [end] en el destino;
  - datos: Combined Data es-ES (textos de salud mental), filtrados a frases cortas
    (<= 20 palabras en español y <= 19 en inglés, para que [start] + 19 + [end]
    quepa en 21 posiciones, igual que sequence_length + 1 del notebook);
  - mask_zero=True en el embedding del codificador: sin máscara, la GRU hacia atrás
    del codificador bidireccional arranca leyendo ceros de relleno;
  - decodificación voraz en lote (mismo algoritmo que generate_translation del
    notebook, pero traduce muchas frases a la vez) para evaluar todo el test.
"""
import re
import string
from collections import Counter

import numpy as np
import pandas as pd
import keras
from keras import layers

COL_ES, COL_EN = "statement_es_es", "statement"
SEQ_LEN = 20

_quitar = string.punctuation + "¿¡«»“”‘’…–—"
_quitar_destino = _quitar.replace("[", "").replace("]", "")
_re_fuente = re.compile(f"[{re.escape(_quitar)}]")
_re_destino = re.compile(f"[{re.escape(_quitar_destino)}]")


_re_apostrofo = re.compile(r"['’‘`´]")


def normalizar_fuente(t):
    t = _re_apostrofo.sub("", str(t).lower())
    return " ".join(_re_fuente.sub(" ", t).split())


def normalizar_destino(t):
    """Minúsculas y sin puntuación; conserva [ y ] para [start]/[end].
    El apóstrofo se borra sin dejar espacio: don't -> dont."""
    t = _re_apostrofo.sub("", str(t).lower())
    return " ".join(_re_destino.sub(" ", t).split())


def cargar_pares(ruta, max_es=20, max_en=19, semilla=42):
    df = pd.read_csv(ruta, usecols=[COL_EN, "status", COL_ES])
    n0 = len(df)
    df = df.dropna(subset=[COL_EN, COL_ES])
    df["es"] = df[COL_ES].map(normalizar_fuente)
    df["en"] = df[COL_EN].map(normalizar_destino)
    df = df[(df.es.str.len() > 0) & (df.en.str.len() > 0)]
    n_validos = len(df)
    df = df[(df.es.str.split().str.len() <= max_es) & (df.en.str.split().str.len() <= max_en)]
    n_cortos = len(df)
    df = df.drop_duplicates(subset=["es", "en"])
    n_dedup = len(df)
    # Pares con "español" idéntico al inglés: no están traducidos (basura, otros
    # idiomas o inglés sin traducir). Enseñarían a copiar e inflarían las métricas.
    df = df[df.es != df.en]
    # Una misma frase en español con dos traducciones distintas se deja (es ruido real).
    df = df.sample(frac=1.0, random_state=semilla).reset_index(drop=True)
    conteo = dict(filas=n0, pares_validos=n_validos, cortos=n_cortos, sin_duplicados=n_dedup,
                  sin_identicos=len(df))
    return df, conteo


def particionar(df, frac=0.15):
    n_val = int(frac * len(df))
    n_tr = len(df) - 2 * n_val
    return df.iloc[:n_tr], df.iloc[n_tr:n_tr + n_val], df.iloc[n_tr + n_val:]


class Vocabulario:
    """Vocabulario por palabras (equivalente a TextVectorization output_mode='int'):
    0 = relleno, 1 = [UNK], resto por frecuencia en entrenamiento."""

    def __init__(self, textos, max_tokens=15000):
        frec = Counter(w for t in textos for w in t.split())
        orden = sorted(frec, key=lambda w: (-frec[w], w))[: max_tokens - 2]
        self.vocab = ["", "[UNK]"] + orden
        self.w2i = {w: i for i, w in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def codificar(self, textos, largo):
        m = np.zeros((len(textos), largo), dtype="int32")
        for f, t in enumerate(textos):
            ids = [self.w2i.get(w, 1) for w in t.split()][:largo]
            m[f, :len(ids)] = ids
        return m


def preparar(tr, va, te, max_tokens=15000):
    v_es = Vocabulario(tr.es, max_tokens)
    v_en = Vocabulario(("[start] " + tr.en + " [end]"), max_tokens)

    def tensores(d):
        x = v_es.codificar(d.es.tolist(), SEQ_LEN)
        y = v_en.codificar(("[start] " + d.en + " [end]").tolist(), SEQ_LEN + 1)
        return {"spanish": x, "english": y[:, :-1]}, y[:, 1:], (y[:, 1:] != 0).astype("float32")

    return v_es, v_en, tensores(tr), tensores(va), tensores(te)


def construir_seq2seq(v_fuente, v_destino, emb=256, oculto=1024, mascara=True):
    fuente = keras.Input(shape=(None,), dtype="int32", name="spanish")
    x = layers.Embedding(v_fuente, emb, mask_zero=mascara)(fuente)
    estado = layers.Bidirectional(layers.GRU(oculto), merge_mode="sum")(x)

    destino = keras.Input(shape=(None,), dtype="int32", name="english")
    x = layers.Embedding(v_destino, emb)(destino)
    x = layers.GRU(oculto, return_sequences=True)(x, initial_state=estado)
    x = layers.Dropout(0.5)(x)
    salida = layers.Dense(v_destino, activation="softmax")(x)
    modelo = keras.Model([fuente, destino], salida)
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   weighted_metrics=["accuracy"])
    return modelo


def traducir_lote(modelo, v_es, v_en, frases_es, lote=256):
    """Decodificación voraz: en el paso i se toma argmax de la posición i."""
    i_start, i_end = v_en.w2i["[start]"], v_en.w2i["[end]"]
    salidas = []
    for a in range(0, len(frases_es), lote):
        x = v_es.codificar(frases_es[a:a + lote], SEQ_LEN)
        n = len(x)
        y = np.zeros((n, SEQ_LEN), dtype="int32")
        y[:, 0] = i_start
        terminado = np.zeros(n, dtype=bool)
        tokens = [[] for _ in range(n)]
        for i in range(SEQ_LEN):
            p = modelo.predict({"spanish": x, "english": y}, verbose=0, batch_size=lote)
            sig = p[:, i, :].argmax(-1)
            for k in range(n):
                if terminado[k]:
                    continue
                if sig[k] == i_end:
                    terminado[k] = True
                else:
                    tokens[k].append(v_en.vocab[sig[k]])
            if i + 1 < SEQ_LEN:
                y[:, i + 1] = sig
            if terminado.all():
                break
        salidas += [" ".join(t) for t in tokens]
    return salidas


# ---------------------------------------------------------------- métricas
from metricas import bleu_corpus, rouge_n, rouge_l, f1_tokens  # implementación propia


def evaluar(hipotesis, referencias):
    """Métricas de corpus (escala 0-100) con la implementación propia de src/metricas.py.
    BLEU: corpus. ROUGE y F1: por frase y promediadas. 'bleu_frase' (solo para la
    distribución por frase) usa sacrebleu.sentence_bleu, que suaviza los ceros."""
    import sacrebleu

    bleu, prec, bp = bleu_corpus(hipotesis, referencias)
    filas = []
    for h, r in zip(hipotesis, referencias):
        p, rc, f = f1_tokens(h, r)
        filas.append(dict(
            bleu_frase=sacrebleu.sentence_bleu(h, [r], tokenize="none").score,
            rouge1=rouge_n(h, r, 1), rouge2=rouge_n(h, r, 2), rougeL=rouge_l(h, r),
            precision_tok=p, recall_tok=rc, f1_tok=f,
        ))
    por_frase = pd.DataFrame(filas)
    resumen = dict(
        BLEU=bleu * 100,
        BLEU_detalle="BLEU = {:.2f} | precisiones 1-4: {} | BP = {:.3f}".format(
            bleu * 100, "/".join(f"{x * 100:.1f}" for x in prec), bp),
        ROUGE1=por_frase.rouge1.mean() * 100, ROUGE2=por_frase.rouge2.mean() * 100,
        ROUGEL=por_frase.rougeL.mean() * 100,
        F1=por_frase.f1_tok.mean() * 100, Precision=por_frase.precision_tok.mean() * 100,
        Recall=por_frase.recall_tok.mean() * 100,
        exactas=float(np.mean([h == r for h, r in zip(hipotesis, referencias)]) * 100),
    )
    return resumen, por_frase
