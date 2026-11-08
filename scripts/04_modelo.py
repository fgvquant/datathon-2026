"""Event study: modelo base (H1) e modelo com score (H2) nos ataques de estimação (2020–2025), com placebo.

Rode da raiz do repositório:
    python scripts/04_modelo.py [--config config/parametros.yaml]

Saída em results/<nome do config>/:
  base_tabela.csv      por moeda: CAR_pre, CR nas velas da tabela, vale e meia-vida (vela), com IC; ataque e placebo
  base_curvas.csv      CR(h) por moeda e vela, com IC; ataque e placebo
  base_curvas.png      gráfico das curvas
  score_b[_SIMULADO].csv, score_curvas[_SIMULADO].csv, score_vale_meia_vida[_SIMULADO].csv, score_curvas[_SIMULADO].png
  score_comum_*        os mesmos quatro arquivos com b comum às 10 moedas
  score_comum_teste.csv  teste de H2: b comum nas velas_teste_score, MQO e Huber, com IC
"""
import argparse
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

from src.modelo import (b_comum_huber, calcula_car, carrega_janelas, inclinacao_comum, intervalo,  # noqa: E402
                        media_ponderada, pesos_bootstrap, regressao_score, vale_meia_vida)

AZUL, LARANJA, TINTA, MUDO, GRADE = "#2a78d6", "#eb6834", "#1f1f1e", "#898781", "#e1e0d9"
RAMPA_SCORE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]  # g = 1..5, claro → escuro


def amostra_estimacao(eventos, cfg):
    """Máscara booleana dos ataques usados na estimação."""
    a = cfg["amostra"]
    return ((eventos.hora_referencia_utc < pd.Timestamp(a["inicio_teste"]))
            & ~eventos.event_id.isin(a["excluir"])).to_numpy()


def resumo_base(r, pesos, moedas, cfg):
    """Tabela longa: moeda × estatística, com estimativa e IC. Velas e vale/meia-vida em vela (h + 1)."""
    m, b = cfg["modelo"], cfg["modelo"]["bootstrap"]
    h_venda = m["vela_venda"] - 1
    cr, cr_rep = r["car"].mean(0), media_ponderada(r["car"], pesos)
    stats = {"CAR_pre": (r["car_pre"].mean(0), media_ponderada(r["car_pre"], pesos))}
    for v in m["velas_tabela"]:
        stats[f"CR_vela_{v:+d}"] = (cr[:, v - 1], cr_rep[:, :, v - 1])
    for nome, i in (("vela_vale", 0), ("vela_meia_vida", 1)):
        stats[nome] = (vale_meia_vida(cr, h_venda, m["minutos_seguidos"])[i] + 1,
                       vale_meia_vida(cr_rep, h_venda, m["minutos_seguidos"])[i] + 1)
    linhas = []
    for nome, (est, rep) in stats.items():
        lo, hi = intervalo(rep, b["nivel_ic"])
        for k, moeda in enumerate(moedas):
            linhas.append({"moeda": moeda, "estatistica": nome, "estimativa": est[k], "ic_inf": lo[k], "ic_sup": hi[k]})
    return pd.DataFrame(linhas), cr, intervalo(cr_rep, b["nivel_ic"])


def curvas_longas(cr, ic, moedas, **extra):
    """CR [I, H] e IC em formato longo (moeda, vela, cr, ic_inf, ic_sup)."""
    I, H = cr.shape
    return pd.DataFrame({"moeda": np.repeat(moedas, H), "vela": np.tile(np.arange(1, H + 1), I),
                         "cr": cr.ravel(), "ic_inf": ic[0].ravel(), "ic_sup": ic[1].ravel(), **extra})


def le_scores(eventos, cfg):
    """Score g por ataque, na ordem de eventos, e se é simulado."""
    caminho = RAIZ / cfg["llm"]["scores"]
    if caminho.exists():
        s = pd.read_csv(caminho).set_index("event_id").gravidade.reindex(eventos.event_id)
        if s.isna().any():
            raise ValueError(f"sem score para {list(s[s.isna()].index)}")
        return s.to_numpy(float), False
    rng = np.random.default_rng(cfg["llm"]["seed_simulados"])
    return rng.integers(1, 6, len(eventos)).astype(float), True


def grafico(curvas, moedas, titulo, arquivo, cor_de, rotulo_de, com_ic=lambda s: True):
    """Pequenos múltiplos: uma CR por moeda, uma linha por série, faixa de IC nas séries com com_ic(serie)."""
    fig, eixos = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    for ax, moeda in zip(eixos.ravel(), moedas):
        for chave, d in curvas[curvas.moeda == moeda].groupby("serie", sort=False):
            if com_ic(chave):
                ax.fill_between(d.vela, d.ic_inf, d.ic_sup, color=cor_de(chave), alpha=0.15, linewidth=0)
            ax.plot(d.vela, d.cr, color=cor_de(chave), linewidth=2, label=rotulo_de(chave))
        ax.axhline(0, color=MUDO, linewidth=1)
        ax.set_title(moeda, color=TINTA, fontsize=11, loc="left")
        ax.grid(color=GRADE, linewidth=0.6)
        ax.tick_params(colors=MUDO, labelsize=9)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRADE)
    for ax in eixos[1]:
        ax.set_xlabel("vela (minutos desde o anúncio)", color=MUDO, fontsize=9)
    for ax in eixos[:, 0]:
        ax.set_ylabel("CR (pb)", color=MUDO, fontsize=9)
    h, l = eixos[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", frameon=False, ncol=len(l))
    fig.suptitle(titulo, x=0.01, ha="left", color=TINTA, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(arquivo, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=RAIZ / "config" / "parametros.yaml")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text(encoding="utf-8"))
    m, b = cfg["modelo"], cfg["modelo"]["bootstrap"]
    saida = RAIZ / "results" / a.config.stem
    saida.mkdir(parents=True, exist_ok=True)

    close, eventos, moedas = carrega_janelas(RAIZ / cfg["amostra"]["precos"])
    close_pl, eventos_pl, moedas_pl = carrega_janelas(RAIZ / cfg["placebo"]["arquivo"])
    assert eventos.event_id.equals(eventos_pl.event_id) and moedas == moedas_pl
    est = amostra_estimacao(eventos, cfg)
    print(f"Estimação: {est.sum()} ataques ({eventos.event_id[est].iloc[0]} a {eventos.event_id[est].iloc[-1]})")

    normal = tuple(cfg["retorno_normal"][k] for k in ("inicio", "fim"))
    r = {k: v[est] for k, v in calcula_car(close, normal, tuple(m["car_pre"])).items()}
    r_pl = {k: v[est] for k, v in calcula_car(close_pl, normal, tuple(m["car_pre"])).items()}
    pesos = pesos_bootstrap(est.sum(), b["replicas"], b["seed"])  # os mesmos sorteios para ataque e placebo

    # ---------- modelo base ----------
    tab, cr, ic = resumo_base(r, pesos, moedas, cfg)
    tab_pl, cr_pl, ic_pl = resumo_base(r_pl, pesos, moedas, cfg)
    tabela = pd.concat([tab.assign(amostra="ataque"), tab_pl.assign(amostra="placebo")])
    tabela.to_csv(saida / "base_tabela.csv", index=False, float_format="%.2f")
    curvas = pd.concat([curvas_longas(cr, ic, moedas, serie="ataque"),
                        curvas_longas(cr_pl, ic_pl, moedas, serie="placebo")])
    curvas.to_csv(saida / "base_curvas.csv", index=False, float_format="%.3f")
    grafico(curvas, moedas, f"CR por moeda: {est.sum()} ataques de 2020–2025 e placebo (IC {b['nivel_ic']:.0%})",
            saida / "base_curvas.png", {"ataque": AZUL, "placebo": LARANJA}.get, str.capitalize)

    fmt = lambda d: d.apply(lambda x: f"{x.estimativa:6.1f} [{x.ic_inf:6.1f};{x.ic_sup:6.1f}]", axis=1)
    for amostra, d in tabela.groupby("amostra"):
        print(f"\n=== Modelo base: {amostra} (pb; vale e meia-vida em vela) ===")
        print(d.assign(v=fmt(d)).pivot(index="moeda", columns="estatistica", values="v")
              [list(dict.fromkeys(d.estatistica))].to_string())

    # ---------- modelo com score ----------
    g_todos, simulado = le_scores(eventos, cfg)
    g = g_todos[est]
    sufixo = "_SIMULADO" if simulado else ""
    a_, b_moeda, gbar = regressao_score(r["car"], g)
    a_rep, b_moeda_rep, gbar_rep = regressao_score(r["car"], g, pesos)

    # "por_moeda": um b_i(h) por moeda (score_*); "comum": b(h) igual nas 10 moedas (score_comum_*), seção 4
    for variante, pre in (("por_moeda", "score"), ("comum", "score_comum")):
        b_, b_rep = ((b_moeda, b_moeda_rep) if variante == "por_moeda"
                     else (inclinacao_comum(b_moeda), inclinacao_comum(b_moeda_rep)))
        lo, hi = intervalo(b_rep, b["nivel_ic"])
        pd.DataFrame([{"moeda": moeda, "vela": v, "b": b_[k, v - 1], "ic_inf": lo[k, v - 1], "ic_sup": hi[k, v - 1]}
                      for k, moeda in enumerate(moedas) for v in m["velas_tabela"]]
                     ).to_csv(saida / f"{pre}_b{sufixo}.csv", index=False, float_format="%.3f")

        curvas_g, vm = [], []
        for nota in range(1, 6):
            crg = a_ + b_ * (nota - gbar)
            crg_rep = a_rep + b_rep * (nota - gbar_rep)[:, None, None]
            curvas_g.append(curvas_longas(crg, intervalo(crg_rep, b["nivel_ic"]), moedas, serie=nota))
            for nome, i in (("vela_vale", 0), ("vela_meia_vida", 1)):
                est_ = vale_meia_vida(crg, m["vela_venda"] - 1, m["minutos_seguidos"])[i] + 1
                lo_, hi_ = intervalo(vale_meia_vida(crg_rep, m["vela_venda"] - 1, m["minutos_seguidos"])[i] + 1,
                                     b["nivel_ic"])
                vm += [{"moeda": moeda, "score": nota, "estatistica": nome, "estimativa": est_[k],
                        "ic_inf": lo_[k], "ic_sup": hi_[k]} for k, moeda in enumerate(moedas)]
        curvas_g = pd.concat(curvas_g)
        curvas_g.to_csv(saida / f"{pre}_curvas{sufixo}.csv", index=False, float_format="%.3f")
        pd.DataFrame(vm).to_csv(saida / f"{pre}_vale_meia_vida{sufixo}.csv", index=False, float_format="%.1f")
        grafico(curvas_g, moedas, f"CR(h; g) por score, b {variante.replace('_', ' ')}"
                f"{' — SCORES SIMULADOS' if simulado else ''} (ḡ = {gbar:.2f}; IC só em g = 1 e 5)",
                saida / f"{pre}_curvas{sufixo}.png",
                lambda s: RAMPA_SCORE[s - 1], lambda s: f"g = {s}", lambda s: s in (1, 5))

    # ---------- teste de H2: b comum em 3 velas, MQO e Huber, IC simples ----------
    b_c, b_c_rep = inclinacao_comum(b_moeda)[0], inclinacao_comum(b_moeda_rep)[:, 0]   # [H] e [R, H]
    teste = []
    for v in m["velas_teste_score"]:
        lo, hi = intervalo(b_c_rep[:, v - 1], b["nivel_ic"])
        teste.append({"vela": v, "metodo": "MQO", "b": b_c[v - 1], "ic_inf": lo, "ic_sup": hi})
        car_v = r["car"][:, :, v - 1]
        lo, hi = intervalo(b_comum_huber(car_v, g, pesos, m["huber_t"]), b["nivel_ic"])
        teste.append({"vela": v, "metodo": "Huber", "b": b_comum_huber(car_v, g, t=m["huber_t"]),
                      "ic_inf": lo, "ic_sup": hi})
    teste = pd.DataFrame(teste)
    teste.to_csv(saida / f"score_comum_teste{sufixo}.csv", index=False, float_format="%.3f")
    print(f"\n=== H2: b comum às moedas (pb por ponto de score){' — SIMULADO' if simulado else ''} ===")
    print(f"{len(m['velas_teste_score'])} testes, IC {b['nivel_ic']:.0%} percentil sem correção")
    print(teste.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    print(f"\nModelo com score{' (SIMULADO)' if simulado else ''}: ḡ = {gbar:.2f}. Saídas em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
