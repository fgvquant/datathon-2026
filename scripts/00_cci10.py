"""Gera o benchmark CCI10 a partir dos CSVs diários do CoinMarketCap.

Rode da raiz do repositório:
    python scripts/00_cci10.py

Entrada: dados/cci10/cmc/      um CSV por moeda (BTC.csv, ETH.csv, ...)
Saída:   dados/cci10/          cci10_nivel_diario.csv, cci10_pesos_rebalanceamento.csv,
                               cci10_pesos_diarios.csv e, se houver, dias_preenchidos.csv

As regras do índice estão em src/cci10.py.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.cci10 import BASE_DATE, build_index, load_all, summary  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=RAIZ / "dados" / "cci10" / "cmc")
    ap.add_argument("--out", type=Path, default=RAIZ / "dados" / "cci10")
    ap.add_argument("--end", default=None, help="data final (AAAA-MM-DD); padrão = último dia comum a todas")
    a = ap.parse_args(argv)

    P, M, report, gaps = load_all(a.data)
    print("Arquivos lidos:")
    for c, d0, d1, n, np_, nm in report:
        print(f"  {c:5s} {d0} -> {d1}  {n:5d} dias  preços NA={np_}  mcap NA={nm}")

    common_end = min(P[c].last_valid_index() for c in P.columns)
    end = pd.Timestamp(a.end) if a.end else common_end
    first_full = max(M[c].first_valid_index() for c in M.columns)
    warmup = (BASE_DATE - first_full).days
    if warmup < 15:
        print(f"AVISO: só {warmup} dias de histórico antes da base; a EWMA pode não ter convergido.")

    level, W, daily_w, Mstar = build_index(P, M, end=end)

    a.out.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"cci10": level.round(4), "ret": level.pct_change()})
    out.index.name = "date"
    out.to_csv(a.out / "cci10_nivel_diario.csv")
    W.to_csv(a.out / "cci10_pesos_rebalanceamento.csv")
    daily_w.round(6).rename_axis("date").to_csv(a.out / "cci10_pesos_diarios.csv")
    filled = (gaps & P.notna()).loc[BASE_DATE:end]
    if filled.values.any():
        filled.stack().loc[lambda s: s].reset_index().to_csv(
            a.out / "dias_preenchidos.csv", index=False, header=["date", "coin", "ffill"])

    s = summary(level)
    print(f"\nResumo {s['inicio']} a {s['fim']}: nível final {s['nivel_final']:.2f} | "
          f"CAGR {s['CAGR']:.2%} | vol anual {s['vol_anual']:.2%} | "
          f"Sharpe (rf=0) {s['sharpe_rf0']:.2f} | max drawdown {s['max_drawdown']:.2%}")
    print("\nÚltimos pesos de rebalanceamento:")
    print(W.iloc[-1].sort_values(ascending=False).to_string())
    return level, W, daily_w


if __name__ == "__main__":
    main()
