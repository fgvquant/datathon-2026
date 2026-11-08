"""Baixa as janelas placebo (placebo A) da Binance spot, velas de 1 min.

Rode da raiz do repositório:
    python scripts/02_baixa_placebo.py

Entrada: dados/eventos/coleta_tweets_ataques.xlsx (timestamp_utc) e config/parametros.yaml (placebo).
Saída:   dados/placebo/minuto_janela_placebo.csv.gz, no formato de minuto_janela_ataques.csv mais a
         coluna `semanas_antes`. Os zips diários ficam em cache em dados/placebo/brutos/ (fora do Git).
"""
import csv
import gzip
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import openpyxl
import requests
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from src.placebo import COLUNAS, FMT, PARES, escolhe_horarios, le_klines, monta_janela  # noqa: E402

URL = "https://data.binance.vision/data/spot/daily/klines/{par}/1m/{par}-1m-{dia}.zip"


def le_ataques(caminho):
    """{event_id: datetime UTC do anúncio} das linhas com timestamp_utc preenchido."""
    ws = openpyxl.load_workbook(caminho, read_only=True)["coleta"]
    linhas = ws.iter_rows(values_only=True)
    cab = next(linhas)
    ie, it = cab.index("event_id"), cab.index("timestamp_utc")
    return {r[ie]: datetime.strptime(str(r[it]), FMT) for r in linhas if r[ie] and r[it]}


def baixa_dia(par, dia, pasta):
    """Bytes do zip diário (com cache em disco)."""
    arq = pasta / par / f"{par}-1m-{dia}.zip"
    if not arq.exists():
        r = requests.get(URL.format(par=par, dia=dia), timeout=60)
        r.raise_for_status()
        arq.parent.mkdir(parents=True, exist_ok=True)
        arq.write_bytes(r.content)
    return arq.read_bytes()


def main():
    cfg = yaml.safe_load((RAIZ / "config" / "parametros.yaml").read_text(encoding="utf-8"))
    pl, jan = cfg["placebo"], cfg["janela"]
    antes, depois = jan["minutos_antes"], jan["minutos_depois"]

    ataques = le_ataques(RAIZ / "dados" / "eventos" / "coleta_tweets_ataques.xlsx")
    escolha = escolhe_horarios(ataques, pl["semanas_antes"], timedelta(hours=pl["distancia_min_horas"]))

    print(f"{len(ataques)} ataques; {len(escolha)} com placebo. Semanas antes: "
          f"{dict(sorted(Counter(k for k, _ in escolha.values()).items()))}")
    for e in sorted(set(ataques) - set(escolha)):
        print(f"  SEM PLACEBO: {e}")
    for e, (k, _) in sorted(escolha.items()):
        if k != pl["semanas_antes"][0]:
            print(f"  {e}: {k} semanas antes (1 semana cruzava outro ataque)")

    pasta = RAIZ / pl["pasta_brutos"]
    dias_de = {e: sorted({(ref + timedelta(minutes=m)).date().isoformat() for m in (-antes, depois)})
               for e, (_, ref) in escolha.items()}
    pedidos = sorted({(par, d) for ds in dias_de.values() for d in ds for par in PARES.values()})
    print(f"Baixando {len(pedidos)} zips diários (com cache)...")
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(lambda pd_: baixa_dia(*pd_, pasta), pedidos))

    saida = RAIZ / pl["arquivo"]
    saida.parent.mkdir(parents=True, exist_ok=True)
    ausentes = 0
    with gzip.open(saida, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUNAS)
        for i, (e, (k, ref)) in enumerate(sorted(escolha.items()), 1):
            for moeda, par in PARES.items():
                velas = {}
                for dia in dias_de[e]:
                    velas.update(le_klines(baixa_dia(par, dia, pasta)))
                for t, hora, v, ausente in monta_janela(velas, ref, antes, depois):
                    ausentes += ausente
                    w.writerow([e, ref.strftime(FMT), moeda, par, t, hora.strftime(FMT), *v,
                                ausente, v[6] == "0", k])
            print(f"\r{i}/{len(escolha)} {e}", end="", flush=True)
    print(f"\nSalvo em {saida.relative_to(RAIZ)} ({saida.stat().st_size / 1e6:.1f} MB); "
          f"velas ausentes (deixadas vazias): {ausentes}")


if __name__ == "__main__":
    main()
