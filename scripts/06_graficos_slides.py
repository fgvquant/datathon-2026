"""Gráficos dos slides (fundo escuro do deck), a partir das tabelas já salvas pelo 04 e pelo 05.

Não reestima nada: lê results/<config>/modelo/tabelas e results/<config>/backtest/tabelas, mais o CCI10 diário e os
horários dos anúncios. Cada figura sai no tamanho exato do espaço no slide (escala 1:1, polegadas, 300 dpi), em
results/<config>/slides/:
  cci10_periodo.png          9,0 × 3,9   CCI10 2020–2026 com estimação × negociação
  cr60_ataque_placebo.png    6,2 × 5,5   dot plot da CR na vela +60 por moeda, ataque × placebo (IC 90% do placebo)
  cr_curvas.png             10,2 × 4,1   CR por moeda nas velas +1..+180, ataque × placebo, painel 2×5
  modelos_comparacao.png     9,2 × 3,1   condicional × vender sempre (linha) e prêmio final dos três (barras)
  estrategia_retorno.png     9,4 × 3,85  estratégia × CCI10 parado, base 100, 2026
  premio_acumulado_s9.png    9,2 × 3,1   prêmio acumulado sobre o CCI10 em 2026 (pb)
  premio_acumulado_s10.png  10,6 × 5,6   o mesmo, com folga acima da linha para as anotações do E473 e do E583
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.estilo import (ALPHA_IC, ATAQUE, CINZA_NOTA, FUNDO, NEGATIVO, NOTA, PLACEBO, SALVA, TEXTO, VALOR, nota,  # noqa: E402
                        pb, rotulo_fim, set_slide_style)

from matplotlib.lines import Line2D  # noqa: E402

CINZA_FRACO = "#8A8A8A"                                         # moedas dentro do IC do placebo
SALVA_TIGHT = {**SALVA, "bbox_inches": "tight", "pad_inches": 0.15}   # pedido para cr60 e cr_curvas
MESES = "Jan Fev Mar Abr Mai Jun Jul Ago Set Out Nov Dez".split()
SERIES = {"ataque": (ATAQUE, "Ataque"), "placebo": (PLACEBO, "Placebo")}


def pct(v):
    return f"{v:+.1f}%".replace("-", "−").replace(".", ",")


def eixo_meses(ax):
    """Ticks mensais no formato Fev/26."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{MESES[mdates.num2date(v).month - 1]}/{mdates.num2date(v):%y}"))


def cr60(tab):
    """CR na vela +60 por moeda (ataque e placebo) e quais moedas ficam fora do IC 90% do placebo."""
    t = tab[tab.estatistica == "CR_vela_+60"].set_index("moeda")
    a, p = t[t.amostra == "ataque"], t[t.amostra == "placebo"].reindex(t[t.amostra == "ataque"].index)
    sig = (a.estimativa < p.ic_inf) | (a.estimativa > p.ic_sup)
    ordem = a.estimativa.sort_values().index          # da maior queda para a menor
    return a.loc[ordem], p.loc[ordem], sig.loc[ordem]


def grafico_cr60(a, p, sig, arquivo):
    """Dot plot da CR em +60. Fora do IC do placebo: ponto cheio, rótulo amarelo em negrito, ligação até a ponta do
    IC e faixa de fundo; dentro do IC: ponto vazado e rótulos em cinza."""
    fig, ax = plt.subplots(figsize=(6.2, 5.5), layout="constrained")
    y = np.arange(len(a))
    for i, m in enumerate(a.index):
        xa = a.estimativa[m]
        if sig[m]:
            ax.axhspan(i - 0.42, i + 0.42, color="#1C1C1C", linewidth=0, zorder=0)
            ponta = p.ic_inf[m] if xa < p.ic_inf[m] else p.ic_sup[m]          # ponta do IC mais perto do ataque
            ax.plot([xa, ponta], [i, i], color=ATAQUE, alpha=0.6, linewidth=1.5, zorder=1)
        ax.annotate(pb(xa).replace("+", ""), (xa, i), xytext=(-10, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=VALOR, color=ATAQUE if sig[m] else CINZA_FRACO,
                    fontweight="bold" if sig[m] else "normal", zorder=4,
                    bbox=None if sig[m] else dict(facecolor=FUNDO, edgecolor="none", pad=1))   # cobre a barra do IC
    ax.errorbar(p.estimativa, y, xerr=[p.estimativa - p.ic_inf, p.ic_sup - p.estimativa], fmt="o", color=PLACEBO,
                markersize=8, elinewidth=1.5, capsize=0, zorder=2)
    s = sig.to_numpy()
    ax.scatter(a.estimativa[s], y[s], s=120, color=ATAQUE, edgecolor=FUNDO, linewidth=0.8, zorder=3)
    ax.scatter(a.estimativa[~s], y[~s], s=90, facecolor=FUNDO, edgecolor=(1, 0.792, 0.345, 0.6), linewidth=1.5,
               zorder=3)
    ax.vlines(0, -0.5, len(a) - 0.4, color=TEXTO, linewidth=0.8, alpha=0.6)
    ax.grid(False)
    ax.set_yticks(y, a.index)
    for rot, m in zip(ax.get_yticklabels(), a.index):
        rot.set_fontweight("bold" if sig[m] else "normal")
        rot.set_color(TEXTO if sig[m] else CINZA_FRACO)
    ax.set_ylim(len(a) - 0.4, -1.4)                     # maior queda no topo, espaço para os rótulos das séries
    ax.set_xlim(a.estimativa.min() - 25, max(p.ic_sup.max(), 0) + 3)
    for x, txt, cor in ((a.estimativa.iloc[0], "Ataque", ATAQUE), (p.estimativa.iloc[0], "Placebo (IC 90%)", PLACEBO)):
        ax.text(x, -1, txt, color=cor, fontsize=VALOR, ha="center", va="center", zorder=4,
                bbox=dict(facecolor=FUNDO, edgecolor="none", pad=1))
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("CR na vela +60 (pb)")
    ax.set_title("Queda até a vela +60: ataque × placebo", fontsize=16, pad=26)
    marcas = [Line2D([], [], linestyle="", marker="o", markersize=10, markerfacecolor=ATAQUE, markeredgecolor=FUNDO,
                     label="fora do IC 90% do placebo"),
              Line2D([], [], linestyle="", marker="o", markersize=9, markerfacecolor=FUNDO,
                     markeredgecolor=(1, 0.792, 0.345, 0.6), markeredgewidth=1.5, label="dentro do IC")]
    ax.legend(handles=marcas, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=NOTA + 1, handletextpad=0.3,
              columnspacing=1.5, borderaxespad=0.2, borderpad=0, labelcolor=TEXTO)
    fig.savefig(arquivo, **SALVA_TIGHT)
    plt.close(fig)


def grafico_curvas(curvas, ordem, sig, arquivo):
    """Painel 2×5. Fora do IC do placebo em +60: título amarelo em negrito; dentro: título cinza, linha do ataque com
    alpha 0,6 e faixas de IC com metade do alpha."""
    fig, eixos = plt.subplots(2, 5, figsize=(10.2, 4.1), sharex=True, sharey=True, layout="constrained")
    fig.get_layout_engine().set(rect=(0, 0.06, 1, 0.94), w_pad=0.02, h_pad=0.02)   # faixa da nota de rodapé
    for k, (ax, m) in enumerate(zip(eixos.flat, ordem)):
        for serie, (cor, nome) in SERIES.items():
            c = curvas[(curvas.moeda == m) & (curvas.serie == serie)]
            ax.fill_between(c.vela, c.ic_inf, c.ic_sup, color=cor, alpha=ALPHA_IC if sig[m] else ALPHA_IC / 2,
                            linewidth=0)
            ax.plot(c.vela, c.cr, color=cor, linewidth=1.5, alpha=0.6 if serie == "ataque" and not sig[m] else 1)
            if k == 4:                                   # rótulos à direita do último painel da 1ª linha
                rotulo_fim(ax, c.vela.iloc[-1], c.cr.iloc[-1], nome, cor, dy=8 if serie == "ataque" else -8)
        ax.axhline(0, color=TEXTO, linewidth=0.8, alpha=0.6)
        ax.set_title(m, color=ATAQUE if sig[m] else CINZA_FRACO, fontweight="bold" if sig[m] else "normal")
        ax.set_xticks([0, 60, 120, 180])
        ax.set_yticks([-50, 0, 50])
        ax.label_outer()
    eixos[1, 2].set_xlabel("vela após o anúncio")
    fig.supylabel("CR (pb)")
    nota(fig, "Título em amarelo: ataque fora do IC 90% do placebo em +60 · faixas: IC 90%")
    fig.savefig(arquivo, **SALVA_TIGHT)
    plt.close(fig)


def grafico_modelos(ops, met, arquivo):
    """Painel da linha (condicional × vender sempre) e barras do prêmio final dos três modelos, proporção 2:1."""
    cores = {"condicional": (ATAQUE, "Condicional"), "base": (PLACEBO, "Vender sempre"),
             "perfeito": ("#6E6E6E", "Perfeita")}
    v = met.set_index("modelo").premio_acumulado_pb.reindex(list(cores))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.2, 3.1), width_ratios=[2, 1], layout="constrained")
    for m in ("condicional", "base"):
        acum = ops[ops.modelo == m].premio_pb.cumsum().to_numpy()   # mesma soma da tabela de métricas
        x = np.arange(len(acum) + 1)
        ax.step(x, np.r_[0, acum], where="pre", color=cores[m][0])
        rotulo_fim(ax, x[-1], acum[-1], f"{pb(v[m])} {cores[m][1]}", cores[m][0])
    ax.axhline(0, color=TEXTO, linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, len(acum))
    ax.set_xlabel("ataque de 2026 (ordem cronológica)")
    ax.set_ylabel("prêmio acumulado (pb)")
    ax.set_title("Condicional × vender sempre")

    bx.barh(range(3), v, color=[c for c, _ in cores.values()], height=0.6)
    for i, val in enumerate(v):
        bx.annotate(pb(val), (max(val, 0), i), xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                    fontsize=VALOR, color=NEGATIVO if val < 0 else TEXTO)
    bx.set_yticks(range(3), [n for _, n in cores.values()])
    bx.invert_yaxis()
    bx.axvline(0, color=TEXTO, linewidth=0.8, alpha=0.6)
    bx.grid(False)
    bx.set_xlim(min(v.min(), 0) - 0.05 * v.max(), 1.6 * v.max())
    bx.set_xticks([0, 200, 400])
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    bx.set_xlabel("prêmio final (pb)")
    bx.set_title("Prêmio final")
    fig.savefig(arquivo, **SALVA)
    plt.close(fig)


def nivel_cci10():
    return pd.read_csv(RAIZ / "dados/cci10/cci10_nivel_diario.csv", parse_dates=["date"]).set_index("date").cci10


def grafico_retorno(estr, ini, arquivo):
    """Estratégia = CCI10 × Π(1 + π_e), π_e somado no dia do ataque; os dois em base 100 na véspera de 2026."""
    nivel = nivel_cci10()
    cci = 100 * nivel[nivel.index >= ini - pd.Timedelta(days=1)] / nivel[nivel.index < ini].iloc[-1]
    fator = pd.Series((1 + estr.premio_pb / 1e4).cumprod().to_numpy(), index=estr.hora_anuncio_utc.dt.normalize())
    est = cci * fator.groupby(level=0).last().reindex(cci.index, method="ffill").fillna(1)
    fig, ax = plt.subplots(figsize=(9.4, 3.85), layout="constrained")
    ax.axhline(100, color=TEXTO, linewidth=0.8, alpha=0.6)
    ax.plot(est.index, est, color=ATAQUE, linewidth=2, zorder=2)
    ax.plot(cci.index, cci, color=PLACEBO, linewidth=1, linestyle="--", zorder=3)
    for s, cor, nome, dy in ((est, ATAQUE, "Estratégia", 8), (cci, PLACEBO, "CCI10 parado", -8)):
        rotulo_fim(ax, s.index[-1], s.iloc[-1], f"{pct(s.iloc[-1] - 100)} {nome}", cor, dx=8, dy=dy)
    lo = ax.get_ylim()[0]
    ax.plot(estr.hora_anuncio_utc, np.full(len(estr), lo), "|", color=TEXTO, alpha=0.7, markersize=10,
            clip_on=False)
    ax.annotate("ataques", (cci.index[0], lo), xytext=(0, 8), textcoords="offset points", fontsize=NOTA,
                color=CINZA_NOTA)
    ax.set_ylim(lo, None)
    eixo_meses(ax)
    ax.set_ylabel(f"nível (base 100 em {ini - pd.Timedelta(days=1):%d/%m/%Y})")
    fig.savefig(arquivo, **SALVA)
    plt.close(fig)


def grafico_cci10(eventos, ini, arquivo):
    nivel = nivel_cci10()
    fig, ax = plt.subplots(figsize=(9.0, 3.9), layout="constrained")
    ax.axvspan(ini, nivel.index[-1], color=ATAQUE, alpha=0.12, linewidth=0)
    ax.plot(nivel.index, nivel, color=TEXTO, linewidth=1.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_yticks([v for v in (100, 200, 500, 1000, 2000, 5000) if nivel.min() / 1.5 < v < nivel.max() * 1.5])
    ax.set_xlim(nivel.index[0] - pd.Timedelta(days=30), nivel.index[-1] + pd.Timedelta(days=30))
    lo, hi = ax.get_ylim()
    teste = (eventos.hora_referencia_utc >= ini).to_numpy()
    for m, cor in ((~teste, PLACEBO), (teste, ATAQUE)):
        ax.plot(eventos.hora_referencia_utc[m], np.full(m.sum(), lo), "|", color=cor, markersize=10, clip_on=False)
    # estimação no alto à esquerda; negociação embaixo à direita (o alto de 2025 é ocupado pela linha)
    for x, y, txt, cor, ha, dx, dy in ((nivel.index[0], hi, f"estimação: {(~teste).sum()} ataques", PLACEBO, "left", 6, -14),
                                       (nivel.index[-1], lo, f"negociação: {teste.sum()} ataques", ATAQUE, "right", -6, 16)):
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=VALOR, color=cor, ha=ha)
    ax.set_ylim(lo, hi)
    rotulo_fim(ax, nivel.index[-1], nivel.iloc[-1], f"{nivel.iloc[-1]:.0f}", TEXTO)
    ax.set_ylabel("nível (base 100 em 01/01/2020, log)")
    fig.savefig(arquivo, **SALVA)
    plt.close(fig)


def grafico_premio(estr, final, ini, arquivo, figsize, ylim):
    """Prêmio acumulado sobre o CCI10 (soma dos prêmios, como na tabela de métricas); final = valor da tabela."""
    t, y = estr.hora_anuncio_utc, estr.premio_pb.cumsum().to_numpy()
    op = estr.n_vendidas.to_numpy() > 0
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.axhline(0, color=PLACEBO, linewidth=1.5)
    ax.annotate("CCI10 parado", (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, 6),
                textcoords="offset points", ha="right", color=PLACEBO, fontsize=VALOR)
    ax.step(np.r_[ini, t], np.r_[0, y], where="post", color=ATAQUE)
    ax.scatter(t[op], y[op], s=40, color=ATAQUE, edgecolor=FUNDO, linewidth=0.8, zorder=3)
    ax.scatter(t[~op], y[~op], s=40, facecolor=FUNDO, edgecolor=ATAQUE, linewidth=1.5, zorder=3)
    rotulo_fim(ax, t.iloc[-1], y[-1], pb(final), ATAQUE, dx=8)
    ax.set_ylim(*ylim)
    ax.set_yticks(range(0, ylim[1] + 1, 20))
    # legenda numa linha só, no topo, fora da área do gráfico
    ax.set_title("●  vendeu alguma moeda      ○  ficou parado", fontsize=VALOR, fontweight="normal", color=ATAQUE)
    eixo_meses(ax)
    ax.set_ylabel("estratégia − CCI10 (pb)")
    fig.savefig(arquivo, **SALVA)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=RAIZ / "config" / "parametros.yaml")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    res = RAIZ / "results" / a.config.stem
    saida = res / "slides"
    saida.mkdir(parents=True, exist_ok=True)
    print("fonte:", set_slide_style())
    ini = pd.Timestamp(cfg["amostra"]["inicio_teste"])
    p = cfg["backtest"]["principal"]

    at, pl, sig = cr60(pd.read_csv(res / "modelo/tabelas/base_tabela.csv"))
    print("fora do IC do placebo em +60:", ", ".join(sig.index[sig]))
    grafico_cr60(at, pl, sig, saida / "cr60_ataque_placebo.png")
    grafico_curvas(pd.read_csv(res / "modelo/tabelas/base_curvas.csv"), at.index, sig, saida / "cr_curvas.png")

    sel = lambda df: df[np.logical_and.reduce([df[k] == v for k, v in p.items()])]  # noqa: E731
    ops = sel(pd.read_csv(res / "backtest/tabelas/operacoes.csv"))
    met = sel(pd.read_csv(res / "backtest/tabelas/metricas.csv"))
    grafico_modelos(ops, met, saida / "modelos_comparacao.png")

    estr = pd.read_csv(res / "backtest/tabelas/estrategia.csv", parse_dates=["hora_anuncio_utc"])
    final = met.set_index("modelo").premio_acumulado_pb["condicional"]
    grafico_retorno(estr, ini, saida / "estrategia_retorno.png")
    grafico_premio(estr, final, ini, saida / "premio_acumulado_s9.png", (9.2, 3.1), (-10, 70))
    grafico_premio(estr, final, ini, saida / "premio_acumulado_s10.png", (10.6, 5.6), (-10, 90))  # folga p/ notas

    ev = pd.read_csv(RAIZ / cfg["amostra"]["precos"], usecols=["event_id", "hora_referencia_utc"]).drop_duplicates()
    ev["hora_referencia_utc"] = pd.to_datetime(ev.hora_referencia_utc)
    grafico_cci10(ev, ini, saida / "cci10_periodo.png")
    print(f"Saídas em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
