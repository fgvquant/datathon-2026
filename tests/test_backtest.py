"""Testes do backtest com dados simulados em que a resposta é conhecida (CLAUDE.md, seção 9)."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import POS, decide, metricas, premio  # noqa: E402
from src.modelo import calcula_car  # noqa: E402

bt = importlib.import_module("scripts.05_backtest")


def test_decide_compara_queda_com_custo():
    cr = np.zeros((1, 180))
    cr[0, 2:] = -30                                    # cai 30 pb depois da vela de venda (h = 1) e fica lá
    assert decide(cr, 1, "vale", 3, 20)[0][0]          # 30 > 20: vende
    assert not decide(cr, 1, "vale", 3, 40)[0][0]      # 30 < 40: não vende


def test_premio_e_teto_de_liquidez():
    close = np.full((1, 361), 100.0)
    close[0, POS + 10:] = 99.0                         # recompra 1% mais barato
    vol = np.full((1, 361), 1e9)
    pi, f = premio(close, vol, np.array([1.0]), np.array([True]), 1, np.array([20]), 0.001, 1e6, 0.1)
    assert np.isclose(pi, 0.01 - 0.002) and f[0] == 1  # 1% − 2 × 0,10%
    vol[:] = 5e6                                       # 10% do volume = US$ 500 mil: metade da carteira
    pi, f = premio(close, vol, np.array([1.0]), np.array([True]), 1, np.array([20]), 0.001, 1e6, 0.1)
    assert np.isclose(f[0], 0.5) and np.isclose(pi, 0.5 * 0.008)
    assert np.isclose(metricas(np.array([1, -3, 1]) / 1e4, np.ones(3, bool))["drawdown_max_pb"], 3)


def test_backtest_nao_ve_o_proprio_ataque_nem_o_futuro():
    rng = np.random.default_rng(0)
    horas = pd.to_datetime(["2025-01-01", "2025-03-01", "2025-06-01", "2025-09-01", "2026-02-01", "2026-05-01"])
    close = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, (6, 2, 361)), -1))
    d = dict(close=close, vol=np.full_like(close, 1e9), close_pl=close.copy(), vol_pl=np.full_like(close, 1e9),
             car=calcula_car(close)["car"], g=np.array([1., 2, 3, 4, 5, 3]), w=np.full((6, 2), 0.5),
             eventos=pd.DataFrame({"event_id": [f"E{i}" for i in range(6)], "hora_referencia_utc": horas}),
             moedas=["BTC", "ETH"])
    cfg = {"amostra": {"inicio_teste": "2026-01-01", "excluir": []}, "janela": {"minutos_depois": 180},
           "modelo": {"minutos_seguidos": 3},
           "backtest": {"fracao_volume": 0.1, "custos": {"zero": {"taxa_pb": 0, "spread_pb": 0, "impacto_pb": 0}}}}
    decisao = lambda x: x.loc[x.modelo == "score", ["event_id", "vendidas"]].to_numpy().tolist()
    for treino in ("fixo", "expansivo"):
        antes = decisao(bt.roda(d, cfg, 2, "vale", treino, 1e6, "zero"))
        d2 = {**d, "car": d["car"].copy()}
        d2["car"][5] = -1e4                            # CAR do último ataque de 2026 vira lixo
        depois = decisao(bt.roda(d2, cfg, 2, "vale", treino, 1e6, "zero"))
        assert antes == depois                         # nenhuma decisão (nem a dele) muda
    # no expansivo, o ataque de fevereiro entra no treino do de maio: mudar a CAR dele muda a curva de maio
    d3 = {**d, "car": d["car"].copy()}
    d3["car"][4] = -1e4
    cr_maio = lambda dd, m: bt.curvas(dd["car"], dd["g"], m, 3.0)[0]
    m_exp = np.array([1, 1, 1, 1, 1, 0], bool)
    assert not np.allclose(cr_maio(d, m_exp), cr_maio(d3, m_exp))
