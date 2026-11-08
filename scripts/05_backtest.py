"""Backtest nos 45 ataques de 2026 (seção 7): modelo com score, modelo base ("vender sempre"), placebo e previsão perfeita.

Rode da raiz do repositório:
    python scripts/05_backtest.py [--config config/parametros.yaml]

Saída em results/<nome do config>/:
  backtest_operacoes.csv  um prêmio por ataque, modelo e combinação de parâmetros (pb), com R_e do CCI10 e da estratégia
  backtest_metricas.csv   métricas por modelo e combinação (pb), Sharpe deflacionado sobre todas as combinações
  backtest_acumulado.png  prêmio acumulado na combinação principal, uma linha por modelo
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


def le_dados(cfg):
    """Preços, volumes, CAR, scores e pesos, todos na ordem de eventos."""
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
    return dict(close=close, vol=vol, close_pl=close_pl, vol_pl=vol_pl, car=car, g=g.to_numpy(float), w=w,
                eventos=eventos, moedas=moedas)


def curvas(car, g, mascara, g_e):
    """CR(·; g_e) com b comum e CR base (a_i), estimadas só nos ataques da máscara. Saem [I, H] em pb."""
    a, b, gbar = regressao_score(car[mascara], g[mascara])
    return a + inclinacao_comum(b) * (g_e - gbar), a


def roda(d, cfg, vela_venda, saida, treino, carteira, custo):
    """Backtest inteiro de uma combinação. Sai DataFrame: ataque × modelo, com prêmio (pb) e moedas vendidas."""
    bt, ev = cfg["backtest"], d["eventos"]
    h_v, seg = vela_venda - 1, cfg["modelo"]["minutos_seguidos"]
    c = bt["custos"][custo]
    lado = (c["taxa_pb"] + c["spread_pb"] / 2 + c["impacto_pb"]) / 1e4
    hora = ev.hora_referencia_utc
    teste = (hora >= pd.Timestamp(cfg["amostra"]["inicio_teste"])).to_numpy()
    est = ~teste & ~ev.event_id.isin(cfg["amostra"]["excluir"]).to_numpy()
    fim = hora + pd.Timedelta(minutes=cfg["janela"]["minutos_depois"])
    linhas = []
    for k in np.flatnonzero(teste):
        # regra 1: o ataque operado nunca entra no próprio treino; no expansivo, só ataques de 2026 já encerrados
        mascara = est | (teste & (fim < hora[k]).to_numpy()) if treino == "expansivo" else est
        assert not mascara[k]
        cr_score, cr_base = curvas(d["car"], d["g"], mascara, d["g"][k])
        close, vol, w = d["close"][k], d["vol"][k], d["w"][k]
        r_cci10 = float((w * (close[:, -1] / close[:, POS + h_v] - 1)).sum())   # da venda até a vela +180
        casos = {"score": (decide(cr_score, h_v, saida, seg, 2e4 * lado), close, vol),
                 "base": (decide(cr_base, h_v, saida, seg, 2e4 * lado), close, vol),
                 "placebo": (decide(cr_score, h_v, saida, seg, 2e4 * lado), d["close_pl"][k], d["vol_pl"][k]),
                 "perfeito": (recompra_perfeita(close, h_v, lado), close, vol)}
        for modelo, ((vende, h_rec), cl, vo) in casos.items():
            pi, f = premio(cl, vo, w, vende, h_v, h_rec, lado, carteira, bt["fracao_volume"])
            linhas.append({"event_id": ev.event_id[k], "modelo": modelo, "premio_pb": 1e4 * pi,
                           "r_cci10_pb": 1e4 * r_cci10, "r_estrategia_pb": 1e4 * (r_cci10 + pi),
                           "n_vendidas": int(vende.sum()), "fracao_executada": float((w * f)[vende].sum()),
                           "vendidas": " ".join(np.array(d["moedas"])[vende])})
    return pd.DataFrame(linhas)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=RAIZ / "config" / "parametros.yaml")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    bt = cfg["backtest"]
    saida = RAIZ / "results" / a.config.stem
    saida.mkdir(parents=True, exist_ok=True)
    d = le_dados(cfg)

    chaves = ["vela_venda", "saida", "treino", "carteira_usd", "custo"]
    grade = list(itertools.product(bt["velas_venda"], bt["saidas"], bt["treinos"], bt["carteiras_usd"], bt["custos"]))
    ops = pd.concat([roda(d, cfg, *comb).assign(**dict(zip(chaves, comb))) for comb in grade])
    ops.to_csv(saida / "backtest_operacoes.csv", index=False, float_format="%.3f")

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
    met.to_csv(saida / "backtest_metricas.csv", index=False, float_format="%.4f")

    p = bt["principal"]
    sel = np.logical_and.reduce([met[k] == p[k] for k in p])
    cols = ["saida", "modelo", "n_operados", "premio_medio_pb", "premio_acumulado_pb", "taxa_acerto", "ir", "t",
            "drawdown_max_pb", "sharpe_deflacionado"]
    print(f"=== Backtest, combinação principal {p} — {len(grade)} combinações no total ===")
    print(met.loc[sel, cols].to_string(index=False, float_format=lambda v: f"{v:8.2f}"))

    fig, eixos = plt.subplots(1, len(bt["saidas"]), figsize=(12, 4.5), sharey=True, squeeze=False)
    for ax, s in zip(eixos[0], bt["saidas"]):
        x = ops[np.logical_and.reduce([ops[k] == p[k] for k in p]) & (ops.saida == s)]
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
    fig.savefig(saida / "backtest_acumulado.png", dpi=150, facecolor="#fcfcfb")
    print(f"\nSaídas em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
