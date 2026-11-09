"""Estilo dos gráficos dos slides (fundo quase preto do deck), em escala 1:1 com o slide.

A figura sai no tamanho exato do espaço no slide (figsize em polegadas, 300 dpi, layout constrained, sem
bbox_inches="tight"), então o tamanho da fonte no código é o tamanho na projeção.
"""

import matplotlib as mpl
import matplotlib.font_manager as fm

FUNDO, TEXTO, GRADE, CINZA_NOTA = "#0F0F0F", "#F7F7F7", "#333333", "#BBBBBB"
ATAQUE, PLACEBO, NEGATIVO = "#FFCA58", "#8CA8B5", "#E45730"   # ataque/estratégia, placebo/benchmark, destaque
ALPHA_IC = 0.2
VALOR, NOTA = 13, 11            # pt: rótulos no fim das linhas e valores; notas de rodapé
SALVA = dict(dpi=300, facecolor=FUNDO)


def set_slide_style():
    """Aplica o estilo escuro do deck no matplotlib (rcParams). Montserrat se instalada; senão Arial/sans-serif."""
    nomes = {f.name for f in fm.fontManager.ttflist}
    fonte = next((n for n in ("Montserrat", "Arial") if n in nomes), "sans-serif")
    mpl.rcParams.update({
        "font.family": fonte, "font.size": 12,
        "figure.facecolor": FUNDO, "axes.facecolor": FUNDO, "savefig.facecolor": FUNDO,
        "text.color": TEXTO, "axes.labelcolor": TEXTO, "axes.edgecolor": TEXTO,
        "xtick.color": TEXTO, "ytick.color": TEXTO, "xtick.labelsize": 12, "ytick.labelsize": 12,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 12,
        "figure.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRADE, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    })
    return fonte


def rotulo_fim(ax, x, y, texto, cor, dx=6, dy=0, **kw):
    """Rótulo direto no fim da linha (x, y em dados), na cor da série, 13 pt."""
    ax.annotate(texto, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", color=cor, fontsize=VALOR,
                annotation_clip=False, **kw)


def nota(fig, texto):
    """Nota de rodapé (11 pt, cinza) no canto inferior esquerdo; reservar a faixa com rect no layout engine."""
    fig.text(0.005, 0.005, texto, fontsize=NOTA, color=CINZA_NOTA, ha="left", va="bottom")


def pb(v, casas=1):
    """Número em pb com vírgula decimal e sinal de menos tipográfico: −35,3 pb."""
    return f"{v:+.{casas}f}".replace("-", "−").replace(".", ",") + " pb"
