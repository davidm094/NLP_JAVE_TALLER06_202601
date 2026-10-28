"""Punto 1: modelo de lenguaje por caracteres (GRU) entrenado con el libro.

Sigue la sección "Training a Shakespeare language model" del notebook del curso
(Deep Learning with Python, 3a ed., cap. 15) con tres cambios:
  1. corpus: Rebelión en la granja (Orwell, trad. M. Souto) en lugar de Shakespeare;
  2. se separa el último 10 % del libro como validación (para medir pérdida y
     perplejidad en texto no visto y parar el entrenamiento a tiempo);
  3. además del muestreo voraz (argmax) del notebook, se genera con temperatura.
La tokenización se hace fuera de tf.data (arreglos numpy) para que el mismo
código corra con backend JAX (Colab) o TensorFlow.
"""
import numpy as np
import keras
from keras import layers


def ventanas(texto, largo):
    """Cortes no solapados de `largo` caracteres (como el notebook)."""
    x = [texto[i:i + largo] for i in range(0, len(texto) - 1, largo)]
    y = [texto[i + 1:i + 1 + largo] for i in range(0, len(texto) - 1, largo)]
    return x, y


class TokenizadorCaracteres:
    """Equivalente a TextVectorization(split='character', standardize=None):
    id 0 = relleno, id 1 = [UNK], el resto ordenado por frecuencia."""

    def __init__(self, texto_entrenamiento):
        from collections import Counter
        frec = Counter(texto_entrenamiento)
        orden = sorted(frec, key=lambda c: (-frec[c], c))
        self.vocab = ["", "[UNK]"] + orden
        self.c2i = {c: i for i, c in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def codificar(self, textos, largo):
        m = np.zeros((len(textos), largo), dtype="int32")
        for f, t in enumerate(textos):
            ids = [self.c2i.get(c, 1) for c in t[:largo]]
            m[f, :len(ids)] = ids
        return m

    def decodificar(self, ids):
        return "".join(self.vocab[i] for i in ids if i > 1)


def preparar(texto, largo=100, frac_val=0.10):
    corte = int(len(texto) * (1 - frac_val))
    tr, va = texto[:corte], texto[corte:]
    tok = TokenizadorCaracteres(tr)
    xtr, ytr = ventanas(tr, largo)
    xva, yva = ventanas(va, largo)
    datos = dict(
        x_tr=tok.codificar(xtr, largo), y_tr=tok.codificar(ytr, largo),
        x_va=tok.codificar(xva, largo), y_va=tok.codificar(yva, largo),
    )
    return tok, datos


def construir_modelo(vocab, largo=100, emb=256, oculto=1024, dropout=0.1):
    entrada = layers.Input(shape=(largo,), dtype="int32", name="token_ids")
    x = layers.Embedding(vocab, emb)(entrada)
    x = layers.GRU(oculto, return_sequences=True)(x)
    x = layers.Dropout(dropout)(x)
    salida = layers.Dense(vocab, activation="softmax")(x)
    modelo = keras.Model(entrada, salida)
    modelo.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                   metrics=["sparse_categorical_accuracy"])
    return modelo


def modelo_generador(modelo, vocab, emb=256, oculto=1024):
    """Misma red pero paso a paso y con estado explícito (igual que el notebook)."""
    entrada = keras.Input(shape=(1,), dtype="int32", name="token_ids")
    estado_in = keras.Input(shape=(oculto,), name="state")
    x = layers.Embedding(vocab, emb)(entrada)
    x, estado_out = layers.GRU(oculto, return_state=True)(x, initial_state=estado_in)
    salida = layers.Dense(vocab, activation="softmax")(x)
    gen = keras.Model(inputs=(entrada, estado_in), outputs=(salida, estado_out))
    gen.set_weights(modelo.get_weights())
    return gen


def generar(gen, tok, prompt, largo=400, temperatura=None, oculto=1024, semilla=0):
    """temperatura=None -> argmax (voraz, como el notebook); si no, muestreo."""
    rng = np.random.default_rng(semilla)
    estado = np.zeros((1, oculto), dtype="float32")
    pred = None
    for c in prompt:
        pred, estado = gen.predict((np.array([[tok.c2i.get(c, 1)]]), estado), verbose=0)
    ids = []
    for _ in range(largo):
        p = np.asarray(pred[0], dtype="float64")
        p[:2] = 0  # nunca generar relleno ni [UNK]
        if temperatura is None:
            sig = int(np.argmax(p))
        else:
            logp = np.log(p + 1e-12) / temperatura
            q = np.exp(logp - logp.max())
            q /= q.sum()
            sig = int(rng.choice(len(q), p=q))
        ids.append(sig)
        pred, estado = gen.predict((np.array([[sig]]), estado), verbose=0)
    return prompt + tok.decodificar(ids)
