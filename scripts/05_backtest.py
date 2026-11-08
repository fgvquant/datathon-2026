"""Backtest nos 45 ataques de 2026 (seção 7): modelo condicional, modelo base ("vender sempre") e previsão perfeita.

Modelo condicional (08/11): retorno bruto desde a vela 0 ~ score + movimento bruto pré-anúncio (`condicional` no
config), estimado em 2020–2025. Substituiu o modelo só com score DEPOIS de ver 2026 (dizer no relatório).
Base = CR média do treino (b = 0, sem pré-anúncio). O placebo fica só nas regressões do modelo (04).

Rode da raiz do repositório:
    python scripts/05_backtest.py [--config config/parametros.yaml]

Saída em results/<nome do config>/backtest/:
  tabelas/trades.csv          combinação principal, estratégia condicional: uma linha por moeda vendida em cada ataque
  tabelas/estrategia.csv      combinação principal, estratégia condicional: prêmio por ataque e retorno acumulado
  tabelas/metricas.csv        métricas por modelo e combinação (pb), Sharpe deflacionado sobre todas as combinações
  tabelas/operacoes.csv       um prêmio por ataque, modelo e combinação de parâmetros (pb), com R_e do CCI10 e da estratégia
  graficos/estrategia_retorno.png   retorno acumulado da estratégia condicional em 2026 (nível diário, base 100)
  graficos/estrategia_vs_cci10.png  estratégia condicional menos CCI10 parado, acumulado em 2026 (pb)
  graficos/cci10_periodo.png        CCI10 de 2020 a 2026, com os períodos de estimação e negociação e os ataques
  graficos/modelos_acumulado.png    prêmio acumulado na combinação principal: condicional, base e previsão perfeita
  graficos/trades_esperado_vs_realizado.png  esperado × realizado em 3 trades vencedores e 3 perdedores (líquido)
  graficos/media_esperado_vs_realizado.png   esperado × realizado em média nos ataques de 2026, por moeda, com IC

Custos: cada cenário de `backtest.custos` é constante (taxa_pb, spread_pb, impacto_pb) ou vem de um arquivo por
ataque, moeda e minuto (`arquivo`, colunas taxa_venda_spot e taxa_compra_spot da base de preços, em fração), com custo
por lado = taxa_venda_spot na venda e taxa_compra_spot na recompra. Na decisão, a taxa da recompra é a observada na vela de venda (regra 2: o futuro não entra).
"""
import argparse
import itertools
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.backtest import POS, decide, metricas, premio, recompra_perfeita, sharpe_deflacionado  # noqa: E402
from src.modelo import calcula_car, carrega_janelas, regressao_condicional, retorno_bruto  # noqa: E402

CORES = {"condicional": "#2a78d6", "base": "#1f1f1e", "perfeito": "#898781"}
ROTULOS = {"condicional": "Condicional (score + pré-anúncio)", "base": "Vender sempre (CR média)",
           "perfeito": "Previsão perfeita"}


def le_corretagem(caminho, eventos, moedas):
    """Custos por lado do arquivo de corretagem, na grade [E, I, 361] de eventos × moedas × velas −180..+180.

    Entra: CSV com event_id, moeda, minuto_relativo, taxa_venda_spot, taxa_compra_spot, em fração por lado.
    Sai: (venda, recompra) em fração por lado.
    """
    df = pd.read_csv(caminho, usecols=["event_id", "moeda", "minuto_relativo", "taxa_venda_spot", "taxa_compra_spot"])
    if df.duplicated(["event_id", "moeda", "minuto_relativo"]).any():
        raise ValueError(f"{caminho}: linhas repetidas")
    grade = pd.MultiIndex.from_product([eventos.event_id, moedas, range(-180, 181)])
    x = df.set_index(["event_id", "moeda", "minuto_relativo"]).reindex(grade)
    if x.isna().any().any():
        raise ValueError(f"{caminho}: {int(x.isna().any(axis=1).sum())} velas sem custo")
    if (x < 0).any().any() or (x.max() > 0.05).any():
        raise ValueError(f"{caminho}: custo negativo ou acima de 5% por lado; a taxa deve estar em fração")
    forma = (len(eventos), len(moedas), 361)
    return x.taxa_venda_spot.to_numpy().reshape(forma), x.taxa_compra_spot.to_numpy().reshape(forma)


def le_dados(cfg):
    """Preços, volumes, CAR, scores, pesos e custos, todos na ordem de eventos."""
    close, eventos, moedas = carrega_janelas(RAIZ / cfg["amostra"]["precos"])
    vol = carrega_janelas(RAIZ / cfg["amostra"]["precos"], "quote_volume")[0]
    normal = tuple(cfg["retorno_normal"][k] for k in ("inicio", "fim"))
    car = calcula_car(close, normal)["car"]
    y, x = retorno_bruto(close, cfg["condicional"]["janelas"])
    g = pd.read_csv(RAIZ / cfg["llm"]["scores"]).set_index("event_id").gravidade.reindex(eventos.event_id)
    if g.isna().any():
        raise ValueError(f"sem score para {list(g[g.isna()].index)}")
    pesos = pd.read_csv(RAIZ / cfg["backtest"]["pesos"], parse_dates=["date"]).set_index("date")[moedas]
    dia = eventos.hora_referencia_utc.dt.normalize()
    w = np.stack([pesos[pesos.index < d].iloc[-1].to_numpy() for d in dia])   # peso do dia anterior
    w = w / w.sum(1, keepdims=True)

    custos = {}
    for nome, c in cfg["backtest"]["custos"].items():
        if "arquivo" in c:
            custos[nome] = le_corretagem(RAIZ / c["arquivo"], eventos, moedas)
        else:
            lado = np.full(close.shape, (c["taxa_pb"] + c["spread_pb"] / 2 + c["impacto_pb"]) / 1e4)
            custos[nome] = (lado, lado)
    return dict(close=close, vol=vol, car=car, y=y, x=x, g=g.to_numpy(float), w=w, eventos=eventos, moedas=moedas,
                custos=custos)


def curva_condicional(d, mascara, k):
    """Curva [I, H] (pb, retorno bruto esperado desde a vela 0) do ataque k pelo modelo condicional.

    Estimada só nos ataques da máscara; preditores do ataque k (score e pré-anúncio, só velas ≤ 0).
    """
    a, coef, zbar = regressao_condicional(d["y"][mascara], d["g"][mascara], d["x"][mascara])
    x = d["x"][k]
    z = np.column_stack([np.full(len(x), d["g"][k]), x])
    return a + (z - zbar) @ coef


def roda(d, cfg, vela_venda, treino, carteira, custo):
    """Backtest inteiro de uma combinação.

    Sai: (ops, trades). ops = ataque × modelo, com prêmio (pb) e moedas vendidas; trades = uma linha por moeda
    vendida pela estratégia condicional, com preços, fração executada, retorno bruto, custo e contribuição (pb).
    """
    bt, ev = cfg["backtest"], d["eventos"]
    h_v, seg = vela_venda - 1, cfg["modelo"]["minutos_seguidos"]
    c_venda, c_rec = d["custos"][custo]
    hora = ev.hora_referencia_utc
    teste = (hora >= pd.Timestamp(cfg["amostra"]["inicio_teste"])).to_numpy()
    est = ~teste & ~ev.event_id.isin(cfg["amostra"]["excluir"]).to_numpy()
    fim = hora + pd.Timedelta(minutes=cfg["janela"]["minutos_depois"])
    linhas, trades = [], []
    for k in np.flatnonzero(teste):
        # regra 1: o ataque operado nunca entra no próprio treino; no expansivo, só ataques de 2026 já encerrados
        mascara = est | (teste & (fim < hora[k]).to_numpy()) if treino == "expansivo" else est
        assert not mascara[k]
        cr_base = d["car"][mascara].mean(0)                 # CR média do treino: b = 0, sem pré-anúncio
        cr_cond = curva_condicional(d, mascara, k)
        close, vol, w = d["close"][k], d["vol"][k], d["w"][k]
        cv, cr = c_venda[k][:, POS + h_v], c_rec[k]
        custo_decisao = 1e4 * (cv + cr[:, POS + h_v])        # pb, ida e volta, só com o que se sabe na venda
        r_cci10 = float((w * (close[:, -1] / close[:, POS + h_v] - 1)).sum())   # da venda até a vela +180
        casos = {"condicional": decide(cr_cond, h_v, seg, custo_decisao),
                 "base": decide(cr_base, h_v, seg, custo_decisao),
                 "perfeito": recompra_perfeita(close, h_v, cv, cr)}
        for modelo, (vende, h_rec) in casos.items():
            p = premio(close, vol, w, vende, h_v, h_rec, cv, cr, carteira, bt["fracao_volume"])
            pi = float(p["contrib"].sum())
            linhas.append({"event_id": ev.event_id[k], "hora_anuncio_utc": hora[k], "modelo": modelo,
                           "premio_pb": 1e4 * pi, "r_cci10_pb": 1e4 * r_cci10, "r_estrategia_pb": 1e4 * (r_cci10 + pi),
                           "n_vendidas": int(vende.sum()), "fracao_executada": float((w * p["f"])[vende].sum()),
                           "vendidas": " ".join(np.array(d["moedas"])[vende])})
            if modelo != "condicional":
                continue
            minuto0 = hora[k].floor("min")
            for i in np.flatnonzero(vende):
                trades.append({
                    "event_id": ev.event_id[k], "hora_anuncio_utc": hora[k], "score": d["g"][k],
                    "moeda": d["moedas"][i], "peso_cci10": w[i],
                    "vela_venda": vela_venda, "vela_recompra": h_rec[i] + 1,
                    "hora_vela_venda_utc": minuto0 + pd.Timedelta(minutes=vela_venda),
                    "hora_vela_recompra_utc": minuto0 + pd.Timedelta(minutes=int(h_rec[i]) + 1),
                    "preco_venda": p["p_venda"][i], "preco_recompra": p["p_rec"][i],
                    "queda_esperada_pb": cr_cond[i, h_v] - cr_cond[i, h_rec[i]],
                    "custo_esperado_pb": custo_decisao[i],
                    "fracao_executada": p["f"][i], "valor_usd": w[i] * p["f"][i] * carteira,
                    "retorno_bruto_pb": 1e4 * p["bruto"][i], "custo_pb": 1e4 * p["custo"][i],
                    "retorno_liquido_pb": 1e4 * (p["bruto"][i] - p["custo"][i]),
                    "contrib_premio_pb": 1e4 * p["contrib"][i]})
    return pd.DataFrame(linhas), pd.DataFrame(trades)


def grafico_vs_cci10(estr, cfg, arquivo, titulo):
    """Retorno da estratégia condicional acima do CCI10 parado, acumulado em 2026 (pb).

    O CCI10 parado é a linha do zero: estratégia = CCI10 × Π(1 + π_e), logo estratégia / CCI10 − 1 = Π(1 + π_e) − 1,
    que é premio_acumulado_pb. Plotar os dois níveis esconde a diferença (pb contra um índice que oscila dezenas de %).
    Um ponto por ataque: cheio se vendeu alguma moeda, vazado se ficou parado.
    """
    t, y = estr.hora_anuncio_utc, estr.premio_acumulado_pb.to_numpy()
    op = estr.n_vendidas.to_numpy() > 0
    ini, azul, fundo = pd.Timestamp(cfg["amostra"]["inicio_teste"]), CORES["condicional"], "#fcfcfb"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(0, color="#898781", linewidth=1.5)
    ax.annotate("CCI10 parado", (1, 0), xycoords=("axes fraction", "data"), xytext=(-4, 5), textcoords="offset points",
                ha="right", color="#5f5e5a", fontsize=9)
    ax.step(np.r_[ini, t], np.r_[0, y], where="post", color=azul, linewidth=2)
    ax.scatter(t[op], y[op], s=40, color=azul, edgecolor=fundo, linewidth=0.8, zorder=3, label="vendeu alguma moeda")
    ax.scatter(t[~op], y[~op], s=40, facecolor=fundo, edgecolor=azul, linewidth=1.5, zorder=3, label="ficou parado")
    ax.annotate(f"{y[-1]:+.2f} pb", (t.iloc[-1], y[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                fontsize=10, color="#1f1f1e")
    lim = 1.3 * max(np.abs(y).max(), 0.5)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("estratégia − CCI10, acumulado (pb; 1 pb = 0,01%)", color="#5f5e5a", fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.tick_params(colors="#5f5e5a", labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#e1e0d9")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color="#1f1f1e")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, facecolor=fundo)
    plt.close(fig)


def estilo(ax):
    """Grade e eixos recessivos, como nos outros gráficos do projeto."""
    ax.grid(color="#e1e0d9", linewidth=0.6)
    ax.tick_params(colors="#5f5e5a", labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#e1e0d9")


def nivel_cci10():
    """Nível diário do CCI10 (base 100 em 01/01/2020)."""
    return pd.read_csv(RAIZ / "dados/cci10/cci10_nivel_diario.csv", parse_dates=["date"]).set_index("date").cci10


def grafico_estrategia(estr, cfg, arquivo, titulo):
    """Nível diário da estratégia condicional em 2026, base 100 na véspera do período de negociação.

    Estratégia = CCI10 × Π(1 + π_e), com π_e somado no dia do ataque. Traços no pé: um por ataque de 2026.
    """
    ini = pd.Timestamp(cfg["amostra"]["inicio_teste"])
    nivel = nivel_cci10()
    nivel = 100 * nivel[nivel.index >= ini - pd.Timedelta(days=1)] / nivel[nivel.index < ini].iloc[-1]
    fator = pd.Series((1 + estr.premio_pb / 1e4).cumprod().to_numpy(), index=estr.hora_anuncio_utc.dt.normalize())
    estrategia = nivel * fator.groupby(level=0).last().reindex(nivel.index, method="ffill").fillna(1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(100, color="#898781", linewidth=1)
    ax.plot(estrategia.index, estrategia, color=CORES["condicional"], linewidth=2)
    ax.annotate(f"{estrategia.iloc[-1] - 100:+.1f}%".replace(".", ","), (estrategia.index[-1], estrategia.iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center", fontsize=10, color="#1f1f1e")
    lo = ax.get_ylim()[0]
    ax.plot(estr.hora_anuncio_utc, np.full(len(estr), lo), "|", color="#5f5e5a", markersize=10, clip_on=False)
    ax.annotate("ataques", (estrategia.index[0], lo), xytext=(0, 8), textcoords="offset points", fontsize=8,
                color="#5f5e5a")
    ax.set_ylim(lo, None)
    ax.set_ylabel(f"nível (base 100 em {ini - pd.Timedelta(days=1):%d/%m/%Y})", color="#5f5e5a", fontsize=9)
    estilo(ax)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color="#1f1f1e")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def grafico_cci10(eventos, cfg, arquivo):
    """CCI10 no período inteiro (escala log), com o período de negociação sombreado e um traço por ataque."""
    nivel = nivel_cci10()
    ini = pd.Timestamp(cfg["amostra"]["inicio_teste"])
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axvspan(ini, nivel.index[-1], color=CORES["condicional"], alpha=0.08, linewidth=0)
    ax.plot(nivel.index, nivel, color="#1f1f1e", linewidth=1.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([v for v in (100, 200, 500, 1000, 2000, 5000) if nivel.min() / 1.5 < v < nivel.max() * 1.5])
    ax.set_xlim(nivel.index[0] - pd.Timedelta(days=30), nivel.index[-1] + pd.Timedelta(days=90))
    lo, hi = ax.get_ylim()
    teste = (eventos.hora_referencia_utc >= ini).to_numpy()
    for m, cor in ((~teste, "#898781"), (teste, CORES["condicional"])):
        ax.plot(eventos.hora_referencia_utc[m], np.full(m.sum(), lo), "|", color=cor, markersize=10, clip_on=False)
    for x, txt in ((nivel.index[0], f"estimação: {(~teste).sum()} ataques"), (ini, f"negociação: {teste.sum()} ataques")):
        ax.annotate(txt, (x, hi), xytext=(4, -14), textcoords="offset points", fontsize=9, color="#5f5e5a")
    ax.set_ylim(lo, hi)
    ax.annotate(f"{nivel.iloc[-1]:.0f}", (nivel.index[-1], nivel.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                va="center", fontsize=10, color="#1f1f1e")
    ax.set_ylabel("nível (base 100 em 01/01/2020, escala log)", color="#5f5e5a", fontsize=9)
    estilo(ax)
    fig.suptitle(f"CCI10, {nivel.index[0]:%m/%Y} a {nivel.index[-1]:%m/%Y}", x=0.01, ha="left", fontsize=12,
                 color="#1f1f1e")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def grafico_trades(trades, d, est, arquivo):
    """Esperado × realizado em 3 trades vencedores e 3 perdedores (retorno líquido), desde a vela de venda (pb).

    esperado  = curva condicional do treino fixo, y(h) − y(h_venda), velas +1 a +180;
    realizado = 1e4 ln(P_t / P_venda) da moeda no ataque operado, velas −180 a +180. Negativo = bom para a venda.
    """
    t = trades.sort_values("retorno_liquido_pb")
    grupos = {"vencedores": t.iloc[::-1].head(3), "perdedores": t.head(3)}
    ev = d["eventos"]
    fig, eixos = plt.subplots(2, 3, figsize=(15, 8.4))
    for linha, (grupo, sel) in zip(eixos, grupos.items()):
        for ax, tr in zip(linha, sel.itertuples()):
            k, i = int(np.flatnonzero(ev.event_id == tr.event_id)[0]), d["moedas"].index(tr.moeda)
            vv, vr = int(tr.vela_venda), int(tr.vela_recompra)
            cr = curva_condicional(d, est, k)[i]
            esperado = cr - cr[vv - 1]
            close = d["close"][k, i]
            velas = np.arange(-180, 181)
            realizado = 1e4 * np.log(close[POS + velas - 1] / close[POS + vv - 1])   # close[POS + h] = vela h + 1
            ax.axvspan(vv, vr, color=CORES["condicional"], alpha=0.07, linewidth=0)
            ax.axhline(0, color="#898781", linewidth=1)
            ax.axvline(0, color="#5f5e5a", linewidth=0.8)
            ax.plot(np.arange(1, 181), esperado, color=CORES["condicional"], linewidth=2, linestyle="--",
                    label="esperado (modelo condicional)")
            ax.plot(velas, realizado, color="#1f1f1e", linewidth=1.4, label="realizado")
            for y, cor in ((esperado[vr - 1], CORES["condicional"]), (realizado[vr + 180], "#1f1f1e")):
                ax.scatter([vr], [y], s=40, color=cor, edgecolor="white", zorder=4)
                ax.annotate(f"{y:+.0f}", (vr, y), xytext=(7, 0), textcoords="offset points", va="center",
                            fontsize=9, color=cor)
            ax.set_title(f"{tr.moeda} · {tr.event_id} · score {int(tr.score)} · venda +{vv}, recompra +{vr}\n"
                         f"líquido {tr.retorno_liquido_pb:+.0f} pb (bruto {tr.retorno_bruto_pb:+.0f}, "
                         f"custo {tr.custo_pb:.0f}; esperado {-tr.queda_esperada_pb:+.0f})",
                         loc="left", fontsize=10, color="#1f1f1e")
            ax.set_xlim(-181, 181)
            estilo(ax)
        linha[0].set_ylabel(f"{grupo}\nvariação desde a venda (pb)", fontsize=10, color="#1f1f1e")
    eixos[0, 0].legend(frameon=False, fontsize=9, loc="lower left")
    for ax in eixos[1]:
        ax.set_xlabel("vela relativa ao anúncio (min; 0 = anúncio)", fontsize=9, color="#5f5e5a")
    fig.suptitle("Esperado × realizado: 3 trades vencedores e 3 perdedores de 2026. Abaixo de 0 = o preço caiu depois "
                 "da venda", x=0.01, ha="left", fontsize=12, color="#1f1f1e")
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def grafico_media(d, est, teste, cfg, arquivo):
    """Esperado × realizado em média nos ataques de 2026, por moeda, retorno bruto desde a vela 0 (pb), com IC.

    esperado  = média, nos ataques de 2026, da curva condicional estimada em 2020–2025 (linear nos preditores,
                então é a curva no preditor médio de 2026); IC reamostrando os ataques do treino;
    realizado = média de y nos ataques de 2026; IC reamostrando os ataques de 2026. Bootstrap por ataque.
    """
    bs = cfg["modelo"]["bootstrap"]
    i_est, i_tes = np.flatnonzero(est), np.flatnonzero(teste)
    z_tes = np.concatenate([np.broadcast_to(d["g"][i_tes, None, None], (*d["x"][i_tes].shape[:2], 1)),
                            d["x"][i_tes]], axis=-1).mean(0)                          # [I, 1+K]

    def esperado(idx):
        a, coef, zbar = regressao_condicional(d["y"][idx], d["g"][idx], d["x"][idx])
        return a + (z_tes - zbar) @ coef

    rng = np.random.default_rng(bs["seed"])
    rep_esp = [esperado(rng.choice(i_est, len(i_est))) for _ in range(bs["replicas"])]
    rep_rea = [d["y"][rng.choice(i_tes, len(i_tes))].mean(0) for _ in range(bs["replicas"])]
    q = [(1 - bs["nivel_ic"]) / 2, (1 + bs["nivel_ic"]) / 2]
    curvas = {"esp": (esperado(i_est), np.quantile(rep_esp, q, axis=0)),
              "rea": (d["y"][i_tes].mean(0), np.quantile(rep_rea, q, axis=0))}
    rot = {"esp": f"esperado: modelo condicional, treino 2020–25 ({len(i_est)} ataques)",
           "rea": f"realizado: média de 2026 ({len(i_tes)} ataques)"}
    cor = {"esp": CORES["condicional"], "rea": "#1f1f1e"}
    velas = np.arange(1, d["y"].shape[2] + 1)
    fig, eixos = plt.subplots(2, 5, figsize=(17, 7.2), sharex=True, sharey=True)
    for i, (ax, moeda) in enumerate(zip(eixos.flat, d["moedas"])):
        ax.axhline(0, color="#898781", linewidth=1)
        ax.axvline(cfg["modelo"]["vela_venda"], color="#5f5e5a", linewidth=0.8, linestyle=":")
        for c, (y, ic) in curvas.items():
            ax.fill_between(velas, ic[0, i], ic[1, i], color=cor[c], alpha=0.13, linewidth=0)
            ax.plot(velas, y[i], color=cor[c], linewidth=2 if c == "esp" else 1.5,
                    linestyle="--" if c == "esp" else "-", label=rot[c])
        ax.set_title(moeda, loc="left", fontsize=10, color="#1f1f1e")
        estilo(ax)
    h, l = eixos[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=2, fontsize=9)
    for ax in eixos[:, 0]:
        ax.set_ylabel("retorno desde a vela 0 (pb)", fontsize=9, color="#1f1f1e")
    for ax in eixos[1]:
        ax.set_xlabel("vela relativa ao anúncio (min)", fontsize=9, color="#5f5e5a")
    fig.suptitle(f"Esperado × realizado em média, por moeda (IC de {bs['nivel_ic']:.0%}, bootstrap por ataque; "
                 f"pontilhado = venda na vela +{cfg['modelo']['vela_venda']})",
                 x=0.01, ha="left", fontsize=12, color="#1f1f1e")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(arquivo, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=RAIZ / "config" / "parametros.yaml")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    bt = cfg["backtest"]
    saida = RAIZ / "results" / a.config.stem / "backtest"
    for sub in ("tabelas", "graficos"):
        (saida / sub).mkdir(parents=True, exist_ok=True)
    d = le_dados(cfg)

    p = bt["principal"]
    chaves = ["vela_venda", "treino", "carteira_usd", "custo"]
    grade = list(itertools.product(bt["velas_venda"], bt["treinos"], bt["carteiras_usd"], d["custos"]))
    ops = []
    for comb in grade:
        o, t = roda(d, cfg, *comb)
        ops.append(o.assign(**dict(zip(chaves, comb))))
        if all(dict(zip(chaves, comb))[k] == v for k, v in p.items()):
            trades = t
    ops = pd.concat(ops)
    ops.to_csv(saida / "tabelas/operacoes.csv", index=False, float_format="%.3f")

    linhas = []
    for comb, x in ops.groupby(chaves, sort=False):
        series = {m: y for m, y in x.groupby("modelo", sort=False)}
        series["condicional_menos_base"] = series["condicional"].assign(
            premio_pb=series["condicional"].premio_pb.to_numpy() - series["base"].premio_pb.to_numpy(),
            n_vendidas=np.maximum(series["condicional"].n_vendidas.to_numpy(), series["base"].n_vendidas.to_numpy()))
        for m, y in series.items():
            linhas.append({**dict(zip(chaves, comb)), "modelo": m,
                           **metricas(y.premio_pb.to_numpy() / 1e4, y.n_vendidas.to_numpy() > 0)})
    met = pd.DataFrame(linhas)
    # Sharpe deflacionado: as tentativas são todas as combinações da estratégia condicional. Não conta o modelo só com
    # score, testado antes e descartado depois de ver 2026: o deflacionado sai otimista (dizer no relatório).
    sr_tent = met.loc[met.modelo == "condicional", "ir"]
    met["sharpe_deflacionado"] = [sharpe_deflacionado(r.ir, r.n_ataques, r.assimetria, r.curtose, sr_tent)
                                  if np.isfinite(r.ir) else np.nan for r in met.itertuples()]
    met.to_csv(saida / "tabelas/metricas.csv", index=False, float_format="%.4f")

    sel = np.logical_and.reduce([met[k] == p[k] for k in p])
    cols = ["modelo", "n_operados", "premio_medio_pb", "premio_acumulado_pb", "taxa_acerto", "ir", "t",
            "drawdown_max_pb", "sharpe_deflacionado"]
    print(f"=== Backtest, combinação principal {p} — {len(grade)} combinações no total ===")
    print(met.loc[sel, cols].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    # ---------- estratégia condicional na combinação principal: trades, acumulado e gráfico contra o CCI10 ----------
    trades.to_csv(saida / "tabelas/trades.csv", index=False, float_format="%.6g")
    principal = ops[np.logical_and.reduce([ops[k] == p[k] for k in p])]
    estr = principal[principal.modelo == "condicional"].drop(columns="modelo").copy()
    # nível no fim do dia do ataque, base 100 na véspera do período de teste; estratégia = CCI10 × Π(1 + π_e)
    nivel = nivel_cci10()
    base = nivel[nivel.index < pd.Timestamp(cfg["amostra"]["inicio_teste"])].iloc[-1]
    estr["premio_acumulado_pb"] = 1e4 * ((1 + estr.premio_pb / 1e4).cumprod() - 1)
    estr["cci10_nivel"] = 100 * nivel.reindex(estr.hora_anuncio_utc.dt.normalize(), method="ffill").to_numpy() / base
    estr["estrategia_nivel"] = estr.cci10_nivel * (1 + estr.premio_acumulado_pb / 1e4)
    estr.to_csv(saida / "tabelas/estrategia.csv", index=False, float_format="%.3f")
    print(f"\nTrades da estratégia condicional: {len(trades)}")
    print(estr[["premio_acumulado_pb", "cci10_nivel", "estrategia_nivel"]].iloc[-1].to_string(
        float_format=lambda v: f"{v:9.2f}"))

    grafico_vs_cci10(estr, cfg, saida / "graficos/estrategia_vs_cci10.png",
                     f"Estratégia condicional contra o CCI10 parado, 2026 ({len(estr.event_id.unique())} ataques)\n"
                     f"venda na vela +{p['vela_venda']}, recompra na meia-vida, "
                     f"treino {p['treino']}, US$ {p['carteira_usd']:,.0f}, custo {p['custo']}")

    grafico_estrategia(estr, cfg, saida / "graficos/estrategia_retorno.png",
                       f"Estratégia condicional: retorno acumulado em 2026 ({len(estr.event_id.unique())} ataques)\n"
                       f"venda na vela +{p['vela_venda']}, recompra na meia-vida, treino {p['treino']}, "
                       f"US$ {p['carteira_usd']:,.0f}, custo {p['custo']}")
    grafico_cci10(d["eventos"], cfg, saida / "graficos/cci10_periodo.png")
    ev = d["eventos"]
    teste = (ev.hora_referencia_utc >= pd.Timestamp(cfg["amostra"]["inicio_teste"])).to_numpy()
    est = ~teste & ~ev.event_id.isin(cfg["amostra"]["excluir"]).to_numpy()     # treino fixo
    grafico_trades(trades, d, est, saida / "graficos/trades_esperado_vs_realizado.png")
    grafico_media(d, est, teste, cfg, saida / "graficos/media_esperado_vs_realizado.png")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, y in principal.groupby("modelo", sort=False):
        ax.plot(np.arange(1, len(y) + 1), y.premio_pb.cumsum(), color=CORES[m], linewidth=2, label=ROTULOS[m])
    ax.axhline(0, color="#898781", linewidth=1)
    ax.set_xlabel("ataque de 2026 (ordem cronológica)")
    ax.set_ylabel("prêmio acumulado sobre o CCI10 (pb)")
    ax.legend(frameon=False, fontsize=9)
    estilo(ax)
    fig.suptitle(f"Backtest 2026: venda na vela +{p['vela_venda']}, recompra na meia-vida, treino {p['treino']}, "
                 f"US$ {p['carteira_usd']:,.0f}, custo {p['custo']}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(saida / "graficos/modelos_acumulado.png", dpi=150, facecolor="#fcfcfb")
    print(f"\nSaídas em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
