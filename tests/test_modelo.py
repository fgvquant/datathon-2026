"""Testes do event study com dados simulados em que a resposta é conhecida (CLAUDE.md, seção 9)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.modelo import calcula_car, pesos_bootstrap, regressao_score, vale_meia_vida  # noqa: E402

VELAS = np.arange(-180, 181)


def precos(retornos_pb):
    """Preços [..., 361] a partir de retornos [..., 360] (pb) das velas −179..+180, começando em 100."""
    lr = np.concatenate([np.zeros(retornos_pb.shape[:-1] + (1,)), retornos_pb / 1e4], axis=-1)
    return 100 * np.exp(np.cumsum(lr, axis=-1))


def retornos_plantados(tendencia=0.3):
    """Tendência de +0,3 pb/min, vazamento de −2 pb/min de −60 a −1 (fora do normal), ataque de −60 pb na vela +1
    e −5 pb/min de +2 a +10."""
    r = np.full(360, tendencia)
    t = VELAS[1:]
    r[(t >= -60) & (t <= -1)] += -2
    r[t == 1] += -60
    r[(t >= 2) & (t <= 10)] += -5
    return r


def test_cr_recupera_ataque_plantado():
    out = calcula_car(precos(retornos_plantados())[None, None, :])
    assert np.isclose(out["mu"][0, 0], 0.3)                # μ só de [−180, −61]: não vê o vazamento
    car = out["car"][0, 0]
    assert np.isclose(car[0], -60) and np.isclose(car[9], -105)   # vela +1 e vela +10 (h = 9)
    assert np.isclose(car[179], -105)
    assert np.isclose(out["car_pre"][0, 0], -20)           # velas −10..−1 com −2 pb/min, vela 0 sem nada


def test_normal_usa_so_a_janela_configurada():
    r = retornos_plantados()
    r[VELAS[1:] == -60] += 1000                            # choque na vela −60: fora de [−180, −61]
    assert np.isclose(calcula_car(precos(r)[None, None, :])["mu"][0, 0], 0.3)
    r[VELAS[1:] == -61] += 1000                            # choque na vela −61: dentro
    assert not np.isclose(calcula_car(precos(r)[None, None, :])["mu"][0, 0], 0.3)


def test_regressao_recupera_b_plantado():
    rng = np.random.default_rng(0)
    g = rng.integers(1, 6, 300).astype(float)
    car = (-10 - 4 * (g - g.mean()))[:, None, None] + rng.normal(0, 3, (300, 2, 5))
    a, b, gbar = regressao_score(car, g)
    assert np.allclose(b, -4, atol=0.3) and np.allclose(a, -10, atol=0.5) and np.isclose(gbar, g.mean())
    a_r, b_r, _ = regressao_score(car, g, pesos_bootstrap(300, 50, 1))
    assert b_r.shape == (50, 2, 5) and np.allclose(b_r.mean(0), -4, atol=0.3)


def test_bootstrap_reamostra_ataques_inteiros():
    c = pesos_bootstrap(10, 7, 3)
    assert c.shape == (7, 10) and (c.sum(1) == 10).all()


def test_vale_e_meia_vida():
    cr = np.zeros(180)
    cr[1:21] = np.linspace(0, -40, 20)                     # cai de 0 (h=1) até −40 (h=20)
    cr[21:] = -40 + np.linspace(0, 30, 159)                # e reverte devagar
    h_vale, h_meia = vale_meia_vida(cr)
    assert h_vale == 20                                    # h=20 e 21 valem −40: o bloco 20..22 tem o máximo mais baixo
    fundo = cr[20:23].max()
    assert (cr[h_meia:h_meia + 3] <= fundo / 2 + 1e-9).all()
    assert not (cr[h_meia - 1:h_meia + 2] <= fundo / 2).all()
    assert vale_meia_vida(np.linspace(0, 10, 180)) == (1, 1)   # sem queda: sai na hora
