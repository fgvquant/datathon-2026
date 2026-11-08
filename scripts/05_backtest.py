"""Backtest nos 45 ataques de 2026 (seção 7): modelo com score, modelo base ("vender sempre"), placebo e previsão perfeita.

Rode da raiz do repositório:
    python scripts/05_backtest.py [--config config/parametros.yaml]

Saída em results/<nome do config>/backtest/:
  tabelas/trades.csv          combinação principal, estratégia com score: uma linha por moeda vendida em cada ataque
  tabelas/estrategia.csv      combinação principal, estratégia com score: prêmio por ataque e retorno acumulado
  tabelas/metricas.csv        métricas por modelo e combinação (pb), Sharpe deflacionado sobre todas as combinações
  tabelas/operacoes.csv       um prêmio por ataque, modelo e combinação de parâmetros (pb), com R_e do CCI10 e da estratégia
  graficos/estrategia_retorno.png   retorno acumulado da estratégia com score em 2026 (nível diário, base 100)
  graficos/estrategia_vs_cci10.png  estratégia com score menos CCI10 parado, acumulado em 2026 (pb)
  graficos/cci10_periodo.png        CCI10 de 2020 a 2026, com os períodos de estimação e negociação e os ataques
  graficos/modelos_acumulado.png    prêmio acumulado na combinação principal: score, base, placebo e previsão perfeita

Custos: cada cenário de `backtest.custos` é constante (taxa_pb, spread_pb, impacto_pb) ou vem de um arquivo por
ataque, moeda e minuto (`arquivo`), com custo por lado = taxa_venda + corretagem na venda e taxa_recompra +
corretagem na recompra. Na decisão, a taxa da recompra é a observada na vela de venda (regra 2: o futuro não entra).
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
from src.modelo import calcula_car, carrega_janelas, inclinacao_comum, regressao_score  # noqa: E402

CORES = {"score": "#2a78d6", "base": "#1f1f1e", "placebo": "#eb6834", "perfeito": "#898781"}
ROTULOS = {"score": "Com score (LLM)", "base": "Vender sempre (b = 0)", "placebo": "Placebo (com score)",
           "perfeito": "Previsão perfeita"}
DIVISOR = {"fracao": 1, "pct": 100, "pb": 1e4}


def le_corretagem(caminho, unidade, eventos, moedas):
    """Custos por lado do arquivo de corretagem, na grade [E, I, 361] de eventos × moedas × velas −180..+180.

    Entra: CSV com event_id, moeda, minuto_relativo, taxa_venda, taxa_recompra, corretagem, na `unidade`
           ("fracao", "pct" ou "pb"). Sai: (venda, recompra) em fração por lado: taxa + corretagem.
    """
    df = pd.read_csv(caminho, usecols=["event_id", "moeda", "minuto_relativo", "taxa_venda", "taxa_recompra",
                                       "corretagem"])
    if df.duplicated(["event_id", "moeda", "minuto_relativo"]).any():
        raise ValueError(f"{caminho}: linhas repetidas")
    grade = pd.MultiIndex.from_product([eventos.event_id, moedas, range(-180, 181)])
    x = df.set_index(["event_id", "moeda", "minuto_relativo"]).reindex(grade) / DIVISOR[unidade]
    if x.isna().any().any():
        raise ValueError(f"{caminho}: {int(x.isna().any(axis=1).sum())} velas sem custo")
    if (x < 0).any().any() or (x.max() > 0.05).any():
        raise ValueError(f"{caminho}: custo negativo ou acima de 5% por lado; confira `unidade` ({unidade})")
    forma = (len(eventos), len(moedas), 361)
    return ((x.taxa_venda + x.corretagem).to_numpy().reshape(forma),
            (x.taxa_recompra + x.corretagem).to_numpy().reshape(forma))


def le_dados(cfg):
    """Preços, volumes, CAR, scores, pesos e custos, todos na ordem de eventos."""
    close, eventos, moedas = carrega_janelas(RAIZ / cfg["amostra"]["precos"])
    vol = carrega_janelas(RAIZ / cfg["amostra"]["precos"], "quote_volume")[0]
    close_pl, ev_pl, _ = carrega_janelas(RAIZ / cfg["placebo"]["arquivo"])
    vol_pl = carrega_janelas(RAIZ / cfg["placebo"]["arquivo"], "quote_volume")[0]
    assert ev_pl.event_id.equals(eventos.event_id)
    normal = tuple(cfg["retorno_normal"][k] for k in ("inicio", "fim"))
    car = calcula_car(close, normal)["car"]
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
            if not (RAIZ / c["arquivo"]).exists():
                print(f"AVISO: {c['arquivo']} não existe; cenário de custo '{nome}' fica de fora")
                continue
            custos[nome] = le_corretagem(RAIZ / c["arquivo"], c["unidade"], eventos, moedas)
        else:
            lado = np.full(close.shape, (c["taxa_pb"] + c["spread_pb"] / 2 + c["impacto_pb"]) / 1e4)
            custos[nome] = (lado, lado)
    return dict(close=close, vol=vol, close_pl=close_pl, vol_pl=vol_pl, car=car, g=g.to_numpy(float), w=w,
                eventos=eventos, moedas=moedas, custos=custos)


def curvas(car, g, mascara, g_e):
    """CR(·; g_e) com b comum e CR base (a_i), estimadas só nos ataques da máscara. Saem [I, H] em pb."""
    a, b, gbar = regressao_score(car[mascara], g[mascara])
    return a + inclinacao_comum(b) * (g_e - gbar), a


def roda(d, cfg, vela_venda, saida, treino, carteira, custo):
    """Backtest inteiro de uma combinação.

    Sai: (ops, trades). ops = ataque × modelo, com prêmio (pb) e moedas vendidas; trades = uma linha por moeda
    vendida pela estratégia com score, com preços, fração executada, retorno bruto, custo e contribuição (pb).
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
        cr_score, cr_base = curvas(d["car"], d["g"], mascara, d["g"][k])
        close, vol, w = d["close"][k], d["vol"][k], d["w"][k]
        cv, cr = c_venda[k][:, POS + h_v], c_rec[k]
        custo_decisao = 1e4 * (cv + cr[:, POS + h_v])        # pb, ida e volta, só com o que se sabe na venda
        r_cci10 = float((w * (close[:, -1] / close[:, POS + h_v] - 1)).sum())   # da venda até a vela +180
        casos = {"score": (decide(cr_score, h_v, saida, seg, custo_decisao), close, vol),
                 "base": (decide(cr_base, h_v, saida, seg, custo_decisao), close, vol),
                 "placebo": (decide(cr_score, h_v, saida, seg, custo_decisao), d["close_pl"][k], d["vol_pl"][k]),
                 "perfeito": (recompra_perfeita(close, h_v, cv, cr), close, vol)}
        for modelo, ((vende, h_rec), cl, vo) in casos.items():
            p = premio(cl, vo, w, vende, h_v, h_rec, cv, cr, carteira, bt["fracao_volume"])
            pi = float(p["contrib"].sum())
            linhas.append({"event_id": ev.event_id[k], "hora_anuncio_utc": hora[k], "modelo": modelo,
                           "premio_pb": 1e4 * pi, "r_cci10_pb": 1e4 * r_cci10, "r_estrategia_pb": 1e4 * (r_cci10 + pi),
                           "n_vendidas": int(vende.sum()), "fracao_executada": float((w * p["f"])[vende].sum()),
                           "vendidas": " ".join(np.array(d["moedas"])[vende])})
            if modelo != "score":
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
                    "queda_esperada_pb": cr_score[i, h_v] - cr_score[i, h_rec[i]],
                    "custo_esperado_pb": custo_decisao[i],
                    "fracao_executada": p["f"][i], "valor_usd": w[i] * p["f"][i] * carteira,
                    "retorno_bruto_pb": 1e4 * p["bruto"][i], "custo_pb": 1e4 * p["custo"][i],
                    "retorno_liquido_pb": 1e4 * (p["bruto"][i] - p["custo"][i]),
                    "contrib_premio_pb": 1e4 * p["contrib"][i]})
    return pd.DataFrame(linhas), pd.DataFrame(trades)


def grafico_vs_cci10(estr, cfg, arquivo, titulo):
    """Retorno da estratégia com score acima do CCI10 parado, acumulado em 2026 (pb).

    O CCI10 parado é a linha do zero: estratégia = CCI10 × Π(1 + π_e), logo estratégia / CCI10 − 1 = Π(1 + π_e) − 1,
    que é premio_acumulado_pb. Plotar os dois níveis esconde a diferença (pb contra um índice que oscila dezenas de %).
    Um ponto por ataque: cheio se vendeu alguma moeda, vazado se ficou parado.
    """
    x = estr[estr.saida == estr.saida.iloc[0]]
    t, y = x.hora_anuncio_utc, x.premio_acumulado_pb.to_numpy()
    op = x.n_vendidas.to_numpy() > 0
    ini, azul, fundo = pd.Timestamp(cfg["amostra"]["inicio_teste"]), CORES["score"], "#fcfcfb"
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
    """Nível diário da estratégia com score em 2026, base 100 na véspera do período de negociação.

    Estratégia = CCI10 × Π(1 + π_e), com π_e somado no dia do ataque. Traços no pé: um por ataque de 2026.
    """
    ini = pd.Timestamp(cfg["amostra"]["inicio_teste"])
    nivel = nivel_cci10()
    nivel = 100 * nivel[nivel.index >= ini - pd.Timedelta(days=1)] / nivel[nivel.index < ini].iloc[-1]
    fator = pd.Series((1 + estr.premio_pb / 1e4).cumprod().to_numpy(), index=estr.hora_anuncio_utc.dt.normalize())
    estrategia = nivel * fator.groupby(level=0).last().reindex(nivel.index, method="ffill").fillna(1)
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.axhline(100, color="#898781", linewidth=1)
    ax.plot(estrategia.index, estrategia, color=CORES["score"], linewidth=2)
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
    ax.axvspan(ini, nivel.index[-1], color=CORES["score"], alpha=0.08, linewidth=0)
    ax.plot(nivel.index, nivel, color="#1f1f1e", linewidth=1.4)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks([v for v in (100, 200, 500, 1000, 2000, 5000) if nivel.min() / 1.5 < v < nivel.max() * 1.5])
    ax.set_xlim(nivel.index[0] - pd.Timedelta(days=30), nivel.index[-1] + pd.Timedelta(days=90))
    lo, hi = ax.get_ylim()
    teste = (eventos.hora_referencia_utc >= ini).to_numpy()
    for m, cor in ((~teste, "#898781"), (teste, CORES["score"])):
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

    p = dict(bt["principal"])
    if p["custo"] not in d["custos"]:
        p["custo"] = next(iter(d["custos"]))
        print(f"AVISO: custo principal '{bt['principal']['custo']}' indisponível; usando '{p['custo']}'")

    chaves = ["vela_venda", "saida", "treino", "carteira_usd", "custo"]
    grade = list(itertools.product(bt["velas_venda"], bt["saidas"], bt["treinos"], bt["carteiras_usd"], d["custos"]))
    ops, trades = [], []
    for comb in grade:
        o, t = roda(d, cfg, *comb)
        ops.append(o.assign(**dict(zip(chaves, comb))))
        if all(dict(zip(chaves, comb))[k] == v for k, v in p.items()):
            trades.append(t.assign(saida=comb[1]))
    ops = pd.concat(ops)
    ops.to_csv(saida / "tabelas/operacoes.csv", index=False, float_format="%.3f")

    linhas = []
    for comb, x in ops.groupby(chaves, sort=False):
        series = {m: y for m, y in x.groupby("modelo", sort=False)}
        series["score_menos_base"] = series["score"].assign(
            premio_pb=series["score"].premio_pb.to_numpy() - series["base"].premio_pb.to_numpy(),
            n_vendidas=np.maximum(series["score"].n_vendidas.to_numpy(), series["base"].n_vendidas.to_numpy()))
        for m, y in series.items():
            linhas.append({**dict(zip(chaves, comb)), "modelo": m,
                           **metricas(y.premio_pb.to_numpy() / 1e4, y.n_vendidas.to_numpy() > 0)})
    met = pd.DataFrame(linhas)
    # Sharpe deflacionado: as tentativas são todas as combinações da estratégia com score (o que se poderia escolher)
    sr_tent = met.loc[met.modelo == "score", "ir"]
    met["sharpe_deflacionado"] = [sharpe_deflacionado(r.ir, r.n_ataques, r.assimetria, r.curtose, sr_tent)
                                  if np.isfinite(r.ir) else np.nan for r in met.itertuples()]
    met.to_csv(saida / "tabelas/metricas.csv", index=False, float_format="%.4f")

    sel = np.logical_and.reduce([met[k] == p[k] for k in p])
    cols = ["saida", "modelo", "n_operados", "premio_medio_pb", "premio_acumulado_pb", "taxa_acerto", "ir", "t",
            "drawdown_max_pb", "sharpe_deflacionado"]
    print(f"=== Backtest, combinação principal {p} — {len(grade)} combinações no total ===")
    print(met.loc[sel, cols].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    # ---------- estratégia com score na combinação principal: trades, acumulado e gráfico contra o CCI10 ----------
    trades = pd.concat(trades)
    trades.to_csv(saida / "tabelas/trades.csv", index=False, float_format="%.6g")
    principal = ops[np.logical_and.reduce([ops[k] == p[k] for k in p])]
    estr = principal[principal.modelo == "score"].drop(columns="modelo").copy()
    # nível no fim do dia do ataque, base 100 na véspera do período de teste; estratégia = CCI10 × Π(1 + π_e)
    nivel = nivel_cci10()
    base = nivel[nivel.index < pd.Timestamp(cfg["amostra"]["inicio_teste"])].iloc[-1]
    estr["premio_acumulado_pb"] = estr.groupby("saida").premio_pb.transform(lambda x: 1e4 * ((1 + x / 1e4).cumprod() - 1))
    estr["cci10_nivel"] = 100 * nivel.reindex(estr.hora_anuncio_utc.dt.normalize(), method="ffill").to_numpy() / base
    estr["estrategia_nivel"] = estr.cci10_nivel * (1 + estr.premio_acumulado_pb / 1e4)
    estr.to_csv(saida / "tabelas/estrategia.csv", index=False, float_format="%.3f")
    print(f"\nTrades da estratégia com score: {len(trades)} "
          f"({', '.join(f'{s}: {n}' for s, n in trades.saida.value_counts().items())})")
    print(estr.groupby("saida")[["premio_acumulado_pb", "cci10_nivel", "estrategia_nivel"]]
          .last().to_string(float_format=lambda v: f"{v:9.2f}"))

    grafico_vs_cci10(estr, cfg, saida / "graficos/estrategia_vs_cci10.png",
                     f"Estratégia com score contra o CCI10 parado, 2026 ({len(estr.event_id.unique())} ataques)\n"
                     f"venda na vela +{p['vela_venda']}, recompra na {estr.saida.iloc[0].replace('_', '-')}, "
                     f"treino {p['treino']}, US$ {p['carteira_usd']:,.0f}, custo {p['custo']}")

    recompra = estr.saida.iloc[0].replace("_", "-")
    grafico_estrategia(estr[estr.saida == estr.saida.iloc[0]], cfg, saida / "graficos/estrategia_retorno.png",
                       f"Estratégia com score: retorno acumulado em 2026 ({len(estr.event_id.unique())} ataques)\n"
                       f"venda na vela +{p['vela_venda']}, recompra na {recompra}, treino {p['treino']}, "
                       f"US$ {p['carteira_usd']:,.0f}, custo {p['custo']}")
    grafico_cci10(d["eventos"], cfg, saida / "graficos/cci10_periodo.png")

    fig, eixos = plt.subplots(1, len(bt["saidas"]), figsize=(12, 4.5), sharey=True, squeeze=False)
    for ax, s in zip(eixos[0], bt["saidas"]):
        x = principal[principal.saida == s]
        for m, y in x.groupby("modelo", sort=False):
            ax.plot(np.arange(1, len(y) + 1), y.premio_pb.cumsum(), color=CORES[m], linewidth=2, label=ROTULOS[m])
        ax.axhline(0, color="#898781", linewidth=1)
        ax.set_title(f"Recompra: {s.replace('_', '-')}", loc="left", fontsize=11)
        ax.set_xlabel("ataque de 2026 (ordem cronológica)")
        ax.grid(color="#e1e0d9", linewidth=0.6)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
    eixos[0, 0].set_ylabel("prêmio acumulado sobre o CCI10 (pb)")
    eixos[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Backtest 2026: venda na vela +{p['vela_venda']}, treino {p['treino']}, "
                 f"US$ {p['carteira_usd']:,.0f}, custo {p['custo']}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(saida / "graficos/modelos_acumulado.png", dpi=150, facecolor="#fcfcfb")
    print(f"\nSaídas em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
