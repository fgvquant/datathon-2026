"""Janelas placebo (placebo A): mesma hora, k semanas antes de cada ataque, sem cruzar outro ataque.

Só usa a biblioteca padrão. O formato de saída é o de dados/precos/minuto_janela_ataques.csv
(ver dados/README.md), mais a coluna `semanas_antes`.
"""
import csv
import io
import zipfile
from datetime import datetime, timedelta, timezone

PARES = {m: f"{m}USDT" for m in ["ADA", "BCH", "BNB", "BTC", "ETH", "LINK", "LTC", "TRX", "XLM", "XRP"]}
CAMPOS_VELA = ["open", "high", "low", "close", "volume", "quote_volume", "n_trades",
               "taker_buy_base", "taker_buy_quote"]
COLUNAS = ["event_id", "hora_referencia_utc", "moeda", "par", "minuto_relativo", "hora_utc",
           *CAMPOS_VELA, "vela_ausente", "sem_negocio", "semanas_antes"]
FMT = "%Y-%m-%d %H:%M:%S"


def escolhe_horarios(ataques, semanas, distancia_min):
    """Escolhe o horário placebo de cada ataque.

    Entra: ataques = {event_id: datetime UTC do anúncio}; semanas = lista de k (semanas antes),
           em ordem de preferência; distancia_min = timedelta mínima entre o placebo e qualquer anúncio.
    Sai:   {event_id: (k, datetime do placebo)}. Ataque sem k válido não aparece.
    """
    tempos = list(ataques.values())
    escolha = {}
    for e, t in ataques.items():
        for k in semanas:
            p = t - timedelta(weeks=k)
            if all(abs(p - u) >= distancia_min for u in tempos):
                escolha[e] = (k, p)
                break
    return escolha


def le_klines(conteudo_zip):
    """Lê um zip diário de klines 1m da Binance (data.binance.vision).

    Sai: {datetime UTC do início da vela: [open, high, low, close, volume, quote_volume, n_trades,
         taker_buy_base, taker_buy_quote]} como strings, sem conversão (igual ao arquivo dos ataques).
    A Binance passou a gravar o spot em microssegundos a partir de 2025; antes, milissegundos.
    """
    with zipfile.ZipFile(io.BytesIO(conteudo_zip)) as z:
        texto = z.read(z.namelist()[0]).decode()
    velas = {}
    for linha in csv.reader(io.StringIO(texto)):
        if not linha or not linha[0].isdigit():  # cabeçalho, se houver
            continue
        ts = int(linha[0])
        seg = ts / 1e6 if ts > 1e14 else ts / 1e3
        hora = datetime.fromtimestamp(seg, tz=timezone.utc).replace(tzinfo=None)
        velas[hora] = [linha[1], linha[2], linha[3], linha[4], linha[5], linha[7], linha[8],
                       linha[9], linha[10]]
    return velas


def monta_janela(velas, referencia, antes=180, depois=180):
    """Recorta a janela de uma moeda em torno de `referencia` (minuto 0 = minuto do horário, para baixo).

    Entra: velas = saída de le_klines (pode juntar vários dias); referencia = datetime UTC.
    Sai:   lista de (minuto_relativo, hora_da_vela, campos, vela_ausente), com antes+depois+1 itens.
    Vela que a Binance não tem: OHLC = último close (ou o próximo, se faltar no início), volumes 0.
    """
    m0 = referencia.replace(second=0, microsecond=0)
    horas = [(t, m0 + timedelta(minutes=t)) for t in range(-antes, depois + 1)]
    # começa com o primeiro close da janela, para o caso de faltar a vela −180
    c = next((velas[h][3] for _, h in horas if h in velas), None)
    if c is None:
        raise ValueError(f"nenhuma vela na janela de {referencia}")
    saida = []
    for t, h in horas:
        v = velas.get(h)
        if v is None:
            saida.append((t, h, [c, c, c, c, "0", "0", "0", "0", "0"], True))
        else:
            c = v[3]
            saida.append((t, h, v, False))
    return saida
