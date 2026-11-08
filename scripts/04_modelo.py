"""Event study: modelo base (H1), modelo com score (H2) e modelo condicional (o do backtest) nos ataques de
estimação (2020–2025), com placebo.

Rode da raiz do repositório:
    python scripts/04_modelo.py [--config config/parametros.yaml]

Saída em results/<nome do config>/modelo/:
  tabelas/base_tabela.csv     por moeda: CAR_pre, CR nas velas da tabela, vale e meia-vida (vela), com IC; ataque e placebo
  tabelas/base_curvas.csv     CR(h) por moeda e vela, com IC; ataque e placebo
  graficos/base_curvas.png    gráfico das curvas
  tabelas/score_comum_b.csv   modelo com score: b comum às 10 moedas por vela, com IC; sufixo _SIMULADO se faltar o score
  tabelas/score_comum_teste.csv  teste de H2: b comum nas velas_teste_score, MQO e Huber, com IC
  tabelas/condicional_coef.csv e graficos/condicional_coef.png
                              modelo condicional (retorno bruto ~ score + pré-anúncio, src.modelo.regressao_condicional):
                              coeficiente de cada preditor no drift desde a vela de venda, y(h) − y(h_venda), com IC;
                              ataque e placebo (no placebo, y e pré-anúncio da janela placebo, g do ataque)
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
                        media_ponderada, pesos_bootstrap, regressao_condicional, regressao_score, retorno_bruto,
                        vale_meia_vida)

AZUL, LARANJA, TINTA, MUDO, GRADE = "#2a78d6", "#eb6834", "#1f1f1e", "#898781", "#e1e0d9"


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


def grafico(curvas, moedas, titulo, arquivo, cor_de, rotulo_de, ncols=5, ylabel="CR (pb)"):
    """Pequenos múltiplos: uma CR por moeda, uma linha por série, com faixa de IC.

    `moedas` são os painéis (coluna `moeda` de curvas); com ncols ≠ 5 os eixos y ficam independentes.
    """
    nl = -(-len(moedas) // ncols)
    fig, eixos = plt.subplots(nl, ncols, figsize=(16, max(4.5, 3.25 * nl)), sharex=True, sharey=ncols == 5,
                              squeeze=False)
    for ax, moeda in zip(eixos.ravel(), moedas):
        for chave, d in curvas[curvas.moeda == moeda].groupby("serie", sort=False):
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
    for ax in eixos[-1]:
        ax.set_xlabel("vela (minutos desde o anúncio)", color=MUDO, fontsize=9)
    for ax in eixos[:, 0]:
        ax.set_ylabel(ylabel, color=MUDO, fontsize=9)
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
    saida = RAIZ / "results" / a.config.stem / "modelo"
    for sub in ("tabelas", "graficos"):
        (saida / sub).mkdir(parents=True, exist_ok=True)

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
    tabela.to_csv(saida / "tabelas/base_tabela.csv", index=False, float_format="%.2f")
    curvas = pd.concat([curvas_longas(cr, ic, moedas, serie="ataque"),
                        curvas_longas(cr_pl, ic_pl, moedas, serie="placebo")])
    curvas.to_csv(saida / "tabelas/base_curvas.csv", index=False, float_format="%.3f")
    grafico(curvas, moedas, f"CR por moeda: {est.sum()} ataques de 2020–2025 e placebo (IC {b['nivel_ic']:.0%})",
            saida / "graficos/base_curvas.png", {"ataque": AZUL, "placebo": LARANJA}.get, str.capitalize)

    fmt = lambda d: d.apply(lambda x: f"{x.estimativa:6.1f} [{x.ic_inf:6.1f};{x.ic_sup:6.1f}]", axis=1)
    for amostra, d in tabela.groupby("amostra"):
        print(f"\n=== Modelo base: {amostra} (pb; vale e meia-vida em vela) ===")
        print(d.assign(v=fmt(d)).pivot(index="moeda", columns="estatistica", values="v")
              [list(dict.fromkeys(d.estatistica))].to_string())

    # ---------- modelo com score ----------
    g_todos, simulado = le_scores(eventos, cfg)
    g = g_todos[est]
    sufixo = "_SIMULADO" if simulado else ""
    b_moeda, gbar = regressao_score(r["car"], g)[1:]
    b_moeda_rep = regressao_score(r["car"], g, pesos)[1]

    # b(h) comum às 10 moedas (seção 4), uma linha por vela
    b_c, b_c_rep = inclinacao_comum(b_moeda)[0], inclinacao_comum(b_moeda_rep)[:, 0]   # [H] e [R, H]
    lo, hi = intervalo(b_c_rep, b["nivel_ic"])
    pd.DataFrame([{"vela": v, "b": b_c[v - 1], "ic_inf": lo[v - 1], "ic_sup": hi[v - 1]} for v in m["velas_tabela"]]
                 ).to_csv(saida / "tabelas" / f"score_comum_b{sufixo}.csv", index=False, float_format="%.3f")

    # ---------- teste de H2: b comum em 3 velas, MQO e Huber, IC simples ----------
    teste = []
    for v in m["velas_teste_score"]:
        lo, hi = intervalo(b_c_rep[:, v - 1], b["nivel_ic"])
        teste.append({"vela": v, "metodo": "MQO", "b": b_c[v - 1], "ic_inf": lo, "ic_sup": hi})
        car_v = r["car"][:, :, v - 1]
        lo, hi = intervalo(b_comum_huber(car_v, g, pesos, m["huber_t"]), b["nivel_ic"])
        teste.append({"vela": v, "metodo": "Huber", "b": b_comum_huber(car_v, g, t=m["huber_t"]),
                      "ic_inf": lo, "ic_sup": hi})
    teste = pd.DataFrame(teste)
    teste.to_csv(saida / "tabelas" / f"score_comum_teste{sufixo}.csv", index=False, float_format="%.3f")
    print(f"\n=== H2: b comum às moedas (pb por ponto de score){' — SIMULADO' if simulado else ''} ===")
    print(f"{len(m['velas_teste_score'])} testes, IC {b['nivel_ic']:.0%} percentil sem correção")
    print(teste.to_string(index=False, float_format=lambda x: f"{x:7.2f}"))

    print(f"\nModelo com score{' (SIMULADO)' if simulado else ''}: ḡ = {gbar:.2f}.")

    # ---------- modelo condicional (o que o backtest usa): retorno bruto ~ score + pré-anúncio, com placebo ----------
    jan, h_v = cfg["condicional"]["janelas"], m["vela_venda"] - 1
    nomes = ["score"] + [f"pre[{i},{f}]" for i, f in jan]
    sorteios = [np.repeat(np.arange(est.sum()), c.astype(int)) for c in pesos]   # mesmos sorteios do bootstrap acima
    linhas, curvas_c = [], []
    for amostra, cl in (("ataque", close), ("placebo", close_pl)):
        y, x = (v[est] for v in retorno_bruto(cl, jan))
        drift = lambda idx: (c := regressao_condicional(y[idx], g[idx], x[idx])[1]) - c[:, [h_v]]  # [1+K, H]
        pont = drift(np.arange(len(y)))
        lo, hi = intervalo(np.array([drift(s) for s in sorteios]), b["nivel_ic"])
        for k, nome in enumerate(nomes):
            linhas += [{"amostra": amostra, "preditor": nome, "vela": v, "coef": pont[k, v - 1],
                        "ic_inf": lo[k, v - 1], "ic_sup": hi[k, v - 1]} for v in m["velas_teste_score"]]
            curvas_c.append(curvas_longas(pont[k:k + 1], (lo[k:k + 1], hi[k:k + 1]), [nome], serie=amostra))
    coef = pd.DataFrame(linhas)
    coef.to_csv(saida / "tabelas" / f"condicional_coef{sufixo}.csv", index=False, float_format="%.4f")
    grafico(pd.concat(curvas_c), nomes, f"Modelo condicional: efeito de cada preditor no drift desde a venda "
            f"(vela +{h_v + 1}), treino e placebo (IC {b['nivel_ic']:.0%}; score em pb/ponto, pré em pb/pb)",
            saida / "graficos" / f"condicional_coef{sufixo}.png", {"ataque": AZUL, "placebo": LARANJA}.get,
            str.capitalize, ncols=len(nomes), ylabel="coeficiente")
    print(f"\n=== Modelo condicional: drift desde a vela +{h_v + 1} ~ score + pré-anúncio (retorno bruto) ===")
    print(coef.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nSaídas em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
