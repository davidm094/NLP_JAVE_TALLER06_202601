"""Figuras del reporte. Todo se lee de results/."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RES, FIGS = RAIZ / "results", RAIZ / "figs"
FIGS.mkdir(exist_ok=True)
AZUL, NARANJA, GRIS = "#2a6fb0", "#d9822b", "#8a8a85"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def _curva(ax, h, clave, titulo, ylab):
    ep = np.arange(1, len(h[clave]) + 1)
    ax.plot(ep, h[clave], "-o", ms=3, color=AZUL, label="entrenamiento")
    ax.plot(ep, h["val_" + clave], "-o", ms=3, color=NARANJA, label="validación")
    mejor = int(np.argmin(h["val_loss"])) + 1
    ax.axvline(mejor, color=GRIS, ls="--", lw=0.8)
    ax.set_title(titulo); ax.set_xlabel("época"); ax.set_ylabel(ylab)
    ax.legend(frameon=False)


def fig_curvas_generacion():
    h = json.load(open(RES / "historia_generacion.json"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    _curva(ax[0], h, "loss", "Pérdida (entropía cruzada)", "nats por carácter")
    _curva(ax[1], h, "sparse_categorical_accuracy", "Exactitud del siguiente carácter", "exactitud")
    fig.tight_layout(); fig.savefig(FIGS / "curvas_generacion.png"); plt.close(fig)


def fig_curvas_traduccion():
    h = json.load(open(RES / "historia_traduccion.json"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    _curva(ax[0], h, "loss", "Pérdida (entropía cruzada)", "nats por palabra")
    _curva(ax[1], h, "accuracy", "Exactitud de la siguiente palabra", "exactitud")
    fig.tight_layout(); fig.savefig(FIGS / "curvas_traduccion.png"); plt.close(fig)


def fig_por_largo():
    d = pd.read_csv(RES / "metricas_por_largo.csv")
    met = ["BLEU", "ROUGE1", "ROUGE2", "ROUGEL", "F1"]
    x = np.arange(len(d)); w = 0.16
    colores = ["#2a6fb0", "#5b9bd5", "#9cc3e6", "#d9822b", "#4d9a5a"]
    fig, ax = plt.subplots(figsize=(8, 2.9))
    for i, m in enumerate(met):
        ax.bar(x + (i - 2) * w, d[m], w, label=m.replace("ROUGE", "ROUGE-"), color=colores[i])
    ax.set_xticks(x, [f"{t} palabras\n(n={n})" for t, n in zip(d.tramo, d.n)])
    ax.set_ylabel("puntaje (0–100)"); ax.set_title("Métricas en test según la longitud de la frase en español")
    ax.set_ylim(0, d[met].values.max() * 1.3)
    ax.legend(ncol=5, frameon=False, fontsize=7.5, loc="upper center")
    fig.tight_layout(); fig.savefig(FIGS / "metricas_por_largo.png"); plt.close(fig)


def fig_distribucion():
    t = pd.read_csv(RES / "traducciones_test.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.6))
    ax[0].hist(t.f1_tok * 100, bins=20, color=AZUL)
    ax[0].set_title("F1 por frase (test)"); ax[0].set_xlabel("F1 (0–100)"); ax[0].set_ylabel("frases")
    ax[1].hist(t.bleu_frase, bins=20, color=NARANJA)
    ax[1].set_title("BLEU por frase (test)"); ax[1].set_xlabel("BLEU (0–100)")
    fig.tight_layout(); fig.savefig(FIGS / "distribucion_frases.png"); plt.close(fig)


def todas():
    fig_curvas_generacion(); fig_curvas_traduccion(); fig_por_largo(); fig_distribucion()


if __name__ == "__main__":
    todas()
