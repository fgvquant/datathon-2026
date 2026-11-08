"""Backtest (CLAUDE.md, seção 7): regra de venda sobre CR(·; g), prêmio por ataque e métricas.

Convenções (as mesmas de src/modelo.py)
  close, vol [I, 361]: preço de fechamento e volume em USDT (quote_volume) das velas −180..+180; vela v → posição v + 180.
  h = vela − 1. Vender na vela de índice h_venda = vender no fechamento da vela h_venda + 1; o mesmo na recompra.
  CR e custos em pb; prêmio π em fração da carteira (multiplique por 1e4 para pb).
"""
import numpy as np
from scipy import stats

from src.modelo import vale_meia_vida

POS = 181  # posição em close da vela de índice h = 0 (vela +1)


def decide(cr, h_venda, seguidos, custo_ida_volta_pb):
    """Regra de venda por moeda, com recompra na meia-vida (decidido em 08/11).

    Entra: cr [I, H] (pb), curva esperada estimada no treino; h_venda (índice); seguidos = minutos seguidos da
           meia-vida; custo de ida e volta em pb.
    Sai:   vende [I] (bool) e h_rec [I] (índice da recompra).
    Vende i se a queda esperada −[CR(h_rec) − CR(h_venda)] supera o custo de ida e volta.
    """
    h_rec = vale_meia_vida(cr, h_venda, seguidos)[1]
    ganho = cr[:, h_venda] - cr[np.arange(len(cr)), h_rec]
    return ganho > custo_ida_volta_pb, h_rec


def recompra_perfeita(close, h_venda, custo_venda, custo_rec):
    """Previsão perfeita: sabendo todos os preços e custos da janela, recompra na vela de maior ganho líquido
    depois da venda e só vende se esse ganho for positivo.

    Entra: close [I, 361]; h_venda (índice); custo_venda [I] e custo_rec [I, 361] (fração por lado, por vela).
    Sai: vende [I], h_rec [I].
    """
    depois = slice(POS + h_venda + 1, None)
    liquido = 1 - close[:, depois] / close[:, [POS + h_venda]] - custo_venda[:, None] - custo_rec[:, depois]
    h_rec = h_venda + 1 + liquido.argmax(1)
    return liquido.max(1) > 0, h_rec


def premio(close, vol, w, vende, h_venda, h_rec, custo_venda, custo_rec, carteira, fracao_volume):
    """Prêmio π_e = Σ_{i vendidas} w_i f_i [(1 − P_rec / P_venda) − custo_venda_i − custo_rec_i], fração da carteira.

    Entra: close, vol [I, 361]; w [I] (pesos do CCI10, somam 1); vende [I]; h_venda; h_rec [I];
           custo_venda [I] e custo_rec [I, 361] (fração por lado; o da recompra é lido na vela de recompra);
           carteira (US$); fracao_volume (máximo da ordem como fração do volume do minuto).
    f_i = parte de w_i que cabe em fracao_volume do volume da vela de venda e da de recompra (1 = posição inteira).
    Sai: dict de arrays [I]: p_venda, p_rec, f, bruto (1 − P_rec/P_venda), custo (ida e volta), contrib
         (contribuição de cada moeda ao prêmio; π_e = contrib.sum()).
    """
    i = np.arange(len(close))
    pv, pr = close[:, POS + h_venda], close[i, POS + h_rec]
    teto = fracao_volume * np.minimum(vol[:, POS + h_venda], vol[i, POS + h_rec])
    # ponytail: ordem executada em uma vela só; fatiar em várias velas aumenta f nas moedas pequenas
    f = np.minimum(1.0, teto / np.maximum(w * carteira, 1e-9))
    bruto, custo = 1 - pr / pv, custo_venda + custo_rec[i, POS + h_rec]
    return dict(p_venda=pv, p_rec=pr, f=f, bruto=bruto, custo=custo, contrib=vende * w * f * (bruto - custo))


def metricas(pi, operou):
    """Métricas de uma série de prêmios por ataque.

    Entra: pi [N] (fração); operou [N] (bool, vendeu alguma moeda). Sai: dict; valores em pb, exceto razões.
    IR = média / desvio do prêmio (o prêmio é o retorno ativo sobre o CCI10), por ataque, sem anualizar.
    Drawdown máximo sobre o prêmio acumulado (soma, pb).
    """
    pb = 1e4 * np.asarray(pi)
    n, dp = len(pb), pb.std(ddof=1)
    acum = pb.cumsum()
    return {"n_ataques": n, "n_operados": int(operou.sum()),
            "premio_medio_pb": pb.mean(), "premio_acumulado_pb": acum[-1],
            "taxa_acerto": (pb[operou] > 0).mean() if operou.any() else np.nan,
            "ir": pb.mean() / dp if dp > 0 else np.nan,
            "t": pb.mean() / (dp / np.sqrt(n)) if dp > 0 else np.nan,
            "drawdown_max_pb": (np.maximum.accumulate(np.r_[0, acum]) - np.r_[0, acum]).max(),
            "assimetria": stats.skew(pb) if dp > 0 else np.nan,
            "curtose": stats.kurtosis(pb, fisher=False) if dp > 0 else np.nan}


def sharpe_deflacionado(sr, n, assimetria, curtose, sr_tentativas):
    """Deflated Sharpe Ratio (Bailey e López de Prado, 2014).

    Entra: sr, assimetria, curtose (não excesso) da série avaliada, por ataque; n = nº de ataques;
           sr_tentativas = SR de todas as variações testadas (define o SR máximo esperado por sorte).
    Sai: probabilidade (0–1) de o SR verdadeiro superar o máximo esperado entre as N tentativas.
    """
    s = np.asarray(sr_tentativas, float)
    s = s[np.isfinite(s)]
    N, g = len(s), np.euler_gamma
    sr0 = np.sqrt(s.var(ddof=1)) * ((1 - g) * stats.norm.ppf(1 - 1 / N) + g * stats.norm.ppf(1 - 1 / (N * np.e)))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(n - 1) / np.sqrt(1 - assimetria * sr + (curtose - 1) / 4 * sr ** 2)))
