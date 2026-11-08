"""
Monta a base por minuto das 10 maiores moedas em volta de cada ataque: 3 horas antes e 3 horas depois do horário da planilha.

Fonte: https://data.binance.vision (arquivo público do spot, pares contra USDT, velas de 1 minuto)
  data/spot/daily/klines/{PAR}/1m/    arquivos diários, que existem desde 2021-03-01
  data/spot/monthly/klines/{PAR}/1m/  arquivos mensais, usados quando não há diário

Entrada: dados/eventos/coleta_tweets_ataques.xlsx (timestamp_utc dos 197 ataques com tweet)

Saídas:
  dados/precos/brutos/{PAR}/*.zip                   zips originais, conferidos pelo .CHECKSUM (SHA-256)
  dados/precos/minuto_janela_ataques.csv          uma linha por ataque x moeda x minuto
  dados/precos/minuto_cobertura_moedas.csv        por moeda: velas esperadas, ausentes e minutos sem negócio
  dados/precos/minuto_falhas.csv                  arquivos que falharam (vazio = tudo certo)

Regras:
  - minuto_relativo 0 é a vela que contém o horário da planilha: ela abre antes do anúncio e fecha depois.
    A vela +1 é a primeira inteira depois do anúncio. A janela vai de -180 a +180 (361 velas).
  - Minuto sem vela continua na base, com vela_ausente = True e preços vazios. Nada é preenchido.
  - Minuto com vela e sem negócio fica com sem_negocio = True.

Usa o curl do sistema. Se cair no meio, é só rodar de novo: o que já foi baixado é pulado.
Como rodar (na pasta do projeto):  python scripts/01_baixa_janelas_ataques.py
"""
import hashlib, io, os, subprocess, zipfile
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # caminhos relativos à raiz do repositório

BASE = 'https://data.binance.vision/data/spot'
PASTA = 'dados/precos/brutos'
MOEDAS = ['BTC', 'ETH', 'XRP', 'BNB', 'ADA', 'LINK', 'XLM', 'TRX', 'LTC', 'BCH']
MIN_ANTES, MIN_DEPOIS = 180, 180
INICIO_DIARIO = pd.Timestamp('2021-03-01')
CAMPOS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'n_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']


def curl(url):
    for _ in range(4):
        r = subprocess.run(['curl', '-s', '-f', '-m', '120', url], capture_output=True)
        if r.returncode == 0:
            return r.stdout
        if r.returncode == 22:          # 404: o arquivo não existe
            return None
    return None


def baixar(par, rel, nome):
    """rel = daily ou monthly. Devolve (caminho, erro)."""
    destino = f'{PASTA}/{par}/{nome}'
    if os.path.exists(destino):
        return destino, None
    url = f'{BASE}/{rel}/klines/{par}/1m/{nome}'
    dado = curl(url)
    soma = curl(url + '.CHECKSUM')
    if dado is None or soma is None:
        return None, 'download'
    if hashlib.sha256(dado).hexdigest() != soma.decode().split()[0]:
        return None, 'checksum'
    with open(destino, 'wb') as f:
        f.write(dado)
    return destino, None


def pega(t):
    """Baixa o arquivo diário e, se não existir, o mensal do mesmo dia."""
    par, dia = t
    if dia >= INICIO_DIARIO:
        cam, erro = baixar(par, 'daily', f'{par}-1m-{dia:%Y-%m-%d}.zip')
        if cam:
            return par, dia, cam, None
    cam, erro = baixar(par, 'monthly', f'{par}-1m-{dia:%Y-%m}.zip')
    return par, dia, cam, erro


def le_zip(arq):
    z = zipfile.ZipFile(arq)
    d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=CAMPOS, dtype=str)
    d = d[d.open_time != 'open_time']
    n = pd.to_numeric(d.open_time)
    d['minuto'] = pd.to_datetime(n.where(n < 1e14, n / 1000), unit='ms')   # microssegundos nos arquivos mais novos
    return d.drop(columns=['open_time', 'close_time', 'ignore'])


# ---- horários
t = pd.read_excel('dados/eventos/coleta_tweets_ataques.xlsx', sheet_name='coleta', dtype=str)
t = t[t.timestamp_utc.notna()][['event_id', 'timestamp_utc']].rename(columns={'timestamp_utc': 'hora_referencia_utc'})
t['ref'] = pd.to_datetime(t.hora_referencia_utc)
t['minuto_zero'] = t.ref.dt.floor('min')
print(f'{len(t)} ataques com horário', flush=True)

# ---- grade: ataque x minuto
grade = []
for r in t.itertuples():
    g = pd.DataFrame({'minuto_relativo': range(-MIN_ANTES, MIN_DEPOIS + 1)})
    g['minuto'] = r.minuto_zero + pd.to_timedelta(g.minuto_relativo, unit='min')
    g['event_id'] = r.event_id
    g['hora_referencia_utc'] = r.hora_referencia_utc
    grade.append(g)
grade = pd.concat(grade, ignore_index=True)
dias = sorted(grade.minuto.dt.floor('D').unique())
necessarios = set(grade.minuto)

# ---- download
tarefas = []
for m in MOEDAS:
    os.makedirs(f'{PASTA}/{m}USDT', exist_ok=True)
    tarefas += [(f'{m}USDT', pd.Timestamp(d)) for d in dias]
print(f'{len(tarefas)} dias de moeda', flush=True)
with ThreadPoolExecutor(max_workers=8) as ex:
    res = list(ex.map(pega, tarefas))
falhas = [(p, f'{d:%Y-%m-%d}', e) for p, d, c, e in res if c is None]
pd.DataFrame(falhas, columns=['par', 'dia', 'erro']).to_csv('dados/precos/minuto_falhas.csv', index=False)
print(f'download pronto, {len(falhas)} falhas', flush=True)

# ---- leitura
velas = []
for par in sorted({p for p, _, _, _ in res}):
    arqs = sorted({c for p, _, c, _ in res if p == par and c})
    d = pd.concat([le_zip(a) for a in arqs], ignore_index=True)
    d = d[d.minuto.isin(necessarios)].drop_duplicates('minuto')
    d.insert(0, 'par', par)
    velas.append(d)
velas = pd.concat(velas, ignore_index=True)

# ---- junta
pares = pd.DataFrame({'moeda': MOEDAS, 'par': [f'{m}USDT' for m in MOEDAS]})
s = grade.merge(pares, how='cross').merge(velas, on=['par', 'minuto'], how='left')
s['vela_ausente'] = s.open.isna()
s['sem_negocio'] = pd.to_numeric(s.n_trades).eq(0)
s = s.rename(columns={'minuto': 'hora_utc'}).sort_values(['event_id', 'moeda', 'minuto_relativo'])
s['hora_utc'] = s.hora_utc.dt.strftime('%Y-%m-%d %H:%M:%S')
ordem = ['event_id', 'hora_referencia_utc', 'moeda', 'par', 'minuto_relativo', 'hora_utc', 'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'n_trades',
         'taker_buy_base', 'taker_buy_quote', 'vela_ausente', 'sem_negocio']
s[ordem].to_csv('dados/precos/minuto_janela_ataques.csv', index=False, encoding='utf-8')

# ---- cobertura
s['com_negocio'] = ~s.vela_ausente & ~s.sem_negocio
c = s.groupby('moeda', sort=False).agg(velas_esperadas=('open', 'size'), velas_ausentes=('vela_ausente', 'sum'), minutos_sem_negocio=('sem_negocio', 'sum'), ataques=('event_id', 'nunique'))
c['parte_com_negocio'] = s.groupby('moeda').com_negocio.mean()
c['primeiro_dia'] = s[~s.vela_ausente].groupby('moeda').hora_utc.min().str[:10]
c = c.reindex(MOEDAS).reset_index()
c.to_csv('dados/precos/minuto_cobertura_moedas.csv', index=False, encoding='utf-8')
print(c.to_string(index=False))
print(f'Pronto: {s.event_id.nunique()} ataques, {len(MOEDAS)} moedas, {len(s)} linhas')
