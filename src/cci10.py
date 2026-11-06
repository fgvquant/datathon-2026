"""
CCI10 — benchmark de 10 criptomoedas com as regras de ponderação do CCi30.

Regras replicadas (Methodology Manual do CCi30, ago/2026):
  * Market cap ajustada M*(T) = média exponencialmente ponderada da market cap diária,
        M*(T) = sum_i M(T-i) e^{-a i} / sum_i e^{-a i},
    com meia-vida de 3 dias  ->  a = ln(2)/3.
  * Peso w_k = sqrt(M*_k) / sum_j sqrt(M*_j), sem teto nem piso, arredondado a 4 casas.
  * Rebalanceamento mensal no 1º dia de cada mês (00:00 UTC).
  * Entre rebalanceamentos, os pesos andam com os preços (buy-and-hold):
        I(t) = I(T0) * sum_k w_k P_k(t) / P_k(T0)
    e o índice é encadeado em cada rebalanceamento (nível contínuo).
  * Base 100 em 2020-01-01.

Diferença em relação ao CCi30: o universo é fixo (as 10 moedas que estiveram em
todas as reconstituições trimestrais de jan/2020 a out/2024), então não há
reconstituição trimestral.

Entrada: uma pasta com um CSV por moeda, nomeado pelo ticker (BTC.csv, ETH.csv...).
Aceita o formato de exportação do CoinGecko (snapshot_at, price, market_cap,
total_volume) ou qualquer CSV com colunas de data, preço e market cap
(nomes reconhecidos abaixo).

Uso: as funções são chamadas por scripts/00_cci10.py.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

COINS = ["BTC", "ETH", "XRP", "BNB", "ADA", "LINK", "XLM", "TRX", "LTC", "BCH"]
HALF_LIFE_DAYS = 3.0
BASE_DATE = pd.Timestamp("2020-01-01")
BASE_VALUE = 100.0
WEIGHT_DECIMALS = 4
MAX_FFILL_DAYS = 3  # CCi30: usa o último preço disponível; >3 dias sem preço é caso extraordinário

DATE_COLS = ["snapshot_at", "date", "Date", "timeOpen", "timestamp", "time", "timestamp_utc"]
PRICE_COLS = ["price", "close", "Close", "Price", "priceUsd"]
MCAP_COLS = ["market_cap", "marketCap", "Market Cap", "market_cap_usd", "mcap"]


def _pick(df, options, what, path):
    for c in options:
        if c in df.columns:
            return c
    raise ValueError(f"{path.name}: não achei coluna de {what}. Colunas: {list(df.columns)}")


def load_coin(path: Path) -> pd.DataFrame:
    sep = ";" if path.read_text(errors="ignore").splitlines()[0].count(";") > 1 else ","
    df = pd.read_csv(path, sep=sep)
    dcol = _pick(df, DATE_COLS, "data", path)
    pcol = _pick(df, PRICE_COLS, "preço", path)
    mcol = _pick(df, MCAP_COLS, "market cap", path)

    d = df[dcol]
    if pd.api.types.is_numeric_dtype(d):  # epoch em segundos ou ms
        unit = "ms" if d.max() > 1e11 else "s"
        dates = pd.to_datetime(d, unit=unit, utc=True)
    else:
        dates = pd.to_datetime(d.astype(str).str.replace(" UTC", "", regex=False), utc=True)

    dates = dates.dt.tz_convert(None)
    if dcol == "timeClose" or (pcol in ("close", "Close") and dcol not in ("snapshot_at",)):
        # Linha datada D com preço de FECHAMENTO de D (CoinMarketCap, Binance klines):
        # o fechamento de D é o valor de 00:00 UTC de D+1, que é o momento de referência
        # do índice (snapshot do CoinGecko / implementação do rebalanceamento do CCi30).
        dates = dates.dt.normalize() + pd.Timedelta(days=1)
    out = pd.DataFrame({
        "date": dates.dt.normalize(),
        "price": pd.to_numeric(df[pcol], errors="coerce"),
        "mcap": pd.to_numeric(df[mcol], errors="coerce"),
    }).dropna(subset=["date"])
    # um ponto por dia: o primeiro snapshot do dia (00:00 UTC no CoinGecko)
    out = out.sort_values("date").groupby("date", as_index=True).first()
    out.loc[out["mcap"] <= 0, "mcap"] = np.nan
    out.loc[out["price"] <= 0, "price"] = np.nan
    return out


def load_all(folder: Path, coins=COINS):
    prices, mcaps, report = {}, {}, []
    for c in coins:
        f = next((p for p in folder.glob("*.csv") if p.stem.upper() == c), None)
        if f is None:
            raise FileNotFoundError(f"Falta o arquivo {c}.csv em {folder}")
        d = load_coin(f)
        prices[c], mcaps[c] = d["price"], d["mcap"]
        report.append((c, d.index.min().date(), d.index.max().date(), len(d),
                       int(d["price"].isna().sum()), int(d["mcap"].isna().sum())))
    P = pd.DataFrame(prices).sort_index()
    M = pd.DataFrame(mcaps).sort_index()
    full = pd.date_range(P.index.min(), P.index.max(), freq="D")
    P, M = P.reindex(full), M.reindex(full)
    gaps = P.isna() | M.isna()
    P = P.ffill(limit=MAX_FFILL_DAYS)
    M = M.ffill(limit=MAX_FFILL_DAYS)
    return P, M, report, gaps


def adjusted_mcap(M: pd.DataFrame) -> pd.DataFrame:
    # adjust=True reproduz a fórmula normalizada sum(M e^{-ai}) / sum(e^{-ai})
    return M.ewm(halflife=HALF_LIFE_DAYS, adjust=True, ignore_na=True).mean()


def rebalance_weights(Mstar_row: pd.Series) -> pd.Series:
    r = np.sqrt(Mstar_row)
    w = (r / r.sum()).round(WEIGHT_DECIMALS)
    # o arredondamento pode deixar a soma 0.9999/1.0001: o resíduo vai para o maior peso
    w[w.idxmax()] += round(1.0 - w.sum(), WEIGHT_DECIMALS)
    return w


def build_index(P: pd.DataFrame, M: pd.DataFrame, base_date=BASE_DATE, end=None):
    Mstar = adjusted_mcap(M)
    end = end or P.index.max()
    days = pd.date_range(base_date, end, freq="D")
    rebal_dates = [d for d in days if d.day == 1]
    if base_date not in rebal_dates:
        rebal_dates = [base_date] + rebal_dates

    level = pd.Series(index=days, dtype=float)
    daily_w = pd.DataFrame(index=days, columns=P.columns, dtype=float)
    weights_log = {}

    I0 = BASE_VALUE
    for k, T0 in enumerate(rebal_dates):
        T1 = rebal_dates[k + 1] if k + 1 < len(rebal_dates) else end + pd.Timedelta(days=1)
        if P.loc[T0].isna().any() or Mstar.loc[T0].isna().any():
            miss = list(P.columns[P.loc[T0].isna() | Mstar.loc[T0].isna()])
            raise ValueError(f"Sem dado em {T0.date()} para {miss}")
        w = rebalance_weights(Mstar.loc[T0])
        weights_log[T0] = w
        span = pd.date_range(T0, T1 - pd.Timedelta(days=1), freq="D")
        rel = P.loc[span] / P.loc[T0]             # P_k(t)/P_k(T0)
        contrib = rel.mul(w, axis=1)             # w_k P_k(t)/P_k(T0)
        lvl = I0 * contrib.sum(axis=1)
        level.loc[span] = lvl
        daily_w.loc[span] = contrib.div(contrib.sum(axis=1), axis=0)
        # nível de abertura do próximo período = nível com os pesos antigos em T1
        if T1 <= end:
            I0 = I0 * float((P.loc[T1] / P.loc[T0] * w).sum())

    W = pd.DataFrame(weights_log).T
    W.index.name = "rebalance_date"
    return level, W, daily_w, Mstar


def summary(level: pd.Series):
    r = level.pct_change().dropna()
    yrs = (level.index[-1] - level.index[0]).days / 365.25
    cagr = (level.iloc[-1] / level.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * math.sqrt(365)
    dd = (level / level.cummax() - 1).min()
    return {"inicio": level.index[0].date(), "fim": level.index[-1].date(),
            "nivel_final": round(level.iloc[-1], 2), "CAGR": cagr,
            "vol_anual": vol, "sharpe_rf0": cagr / vol if vol else np.nan,
            "max_drawdown": dd}
