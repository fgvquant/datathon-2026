"""
Dá um score de 1 a 5 para o primeiro anúncio de cada um dos 197 ataques, usando um LLM.

Entradas:
  dados/eventos/coleta_tweets_ataques.xlsx  texto_original de cada ataque
  dados/llm/regua_llm.txt                   texto com as regras do score, enviado como instrução ao modelo
  config/parametros.yaml                    llm.rodadas, llm.modelo e amostra.inicio_teste

Saídas:
  dados/llm/respostas.jsonl   uma linha por chamada, só acrescentada, para auditoria
  dados/llm/scores.csv        uma linha por ataque: event_id, gravidade (mediana das rodadas), n_rodadas,
                              notas de cada rodada, amplitude, modelo e regua_sha. É o que o modelo lê.

Regras:
  - Entram os 197 ataques com notícia.
  - O modelo recebe só a régua e o texto_original. Nenhuma outra coluna é enviada.
  - Nenhuma ferramenta é declarada, então o modelo não acessa a internet.
  - Cada notícia roda RODADAS vezes e o score é a mediana.
  - Uma notícia com alguma rodada sem nota fica com score vazio, para conferência.
  - Se parar no meio, é só rodar de novo: o que já foi respondido é pulado.
  - Sem chamadas pendentes, só regrava dados/llm/scores.csv a partir de respostas.jsonl (não precisa de API).
  - --limite sorteia notícias só da calibração, para ajustar a régua sem olhar o teste. A rodada final dá score aos 197.

Antes de rodar: pip install anthropic   e defina a variável de ambiente ANTHROPIC_API_KEY.

Como rodar (na raiz do repositório):
  python scripts/03_score_llm.py --sem-api       mostra o que seria enviado, sem chamar nem pagar nada
  python scripts/03_score_llm.py --limite 10     testa com 10 notícias do período de calibração
  python scripts/03_score_llm.py                 roda os 197
"""
import argparse, hashlib, json, os
from datetime import datetime, timezone
from typing import Literal
import pandas as pd
import yaml

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # caminhos relativos à raiz do repositório
CFG = yaml.safe_load(open('config/parametros.yaml', encoding='utf-8'))

ARQ_NOTICIAS = 'dados/eventos/coleta_tweets_ataques.xlsx'
ARQ_REGUA = 'dados/llm/regua_llm.txt'
ARQ_RESPOSTAS = 'dados/llm/respostas.jsonl'
ARQ_SCORES = CFG['llm']['scores']
MODELO = CFG['llm']['modelo']
RODADAS = CFG['llm']['rodadas']
MAX_TOKENS = 2000
CORTE = pd.Timestamp(CFG['amostra']['inicio_teste'])   # calibração antes, teste (2026) a partir daqui


def ler_feitas(regua_sha):
    feitas = {}
    if os.path.exists(ARQ_RESPOSTAS):
        with open(ARQ_RESPOSTAS, encoding='utf-8') as f:
            for linha in f:
                r = json.loads(linha)
                if r['modelo'] == MODELO and r['regua_sha'] == regua_sha:
                    feitas[(r['event_id'], r.get('rodada', 1))] = r   # a 1ª leva (07/11) não gravou a rodada
    return feitas


def gravar_scores(feitas, ids, regua_sha):
    linhas = []
    for e in ids:
        notas = [feitas.get((e, i), {}).get('score') for i in range(1, RODADAS + 1)]
        validas = [n for n in notas if n is not None]
        completo = len(validas) == RODADAS
        linha = {'event_id': e, 'gravidade': int(pd.Series(validas).median()) if completo else None, 'n_rodadas': len(validas),
                 **{f'nota_{i}': n for i, n in enumerate(notas, 1)}}
        linha.update(amplitude=max(validas) - min(validas) if completo else None, modelo=MODELO, regua_sha=regua_sha)
        linhas.append(linha)
    pd.DataFrame(linhas).astype({'gravidade': 'Int64', 'amplitude': 'Int64'}).to_csv(ARQ_SCORES, index=False, encoding='utf-8')


a = argparse.ArgumentParser()
a.add_argument('--sem-api', action='store_true', help='mostra o que seria enviado e não chama o modelo')
a.add_argument('--limite', type=int, help='sorteia N notícias do período de calibração')
args = a.parse_args()

regua = open(ARQ_REGUA, encoding='utf-8').read()
regua_sha = hashlib.sha256(regua.encode()).hexdigest()[:12]
n = pd.read_excel(ARQ_NOTICIAS, sheet_name='coleta', dtype=str)
n = n[n.texto_original.notna()]
todos = list(n.event_id)   # scores.csv sempre traz todos; com --limite os não sorteados ficam vazios
if args.limite:
    n['data'] = pd.to_datetime(n.timestamp_utc)
    n = n[n.data < CORTE].sample(args.limite, random_state=1)
textos = dict(zip(n.event_id, n.texto_original))
print(f'{len(textos)} notícias, {RODADAS} rodadas, modelo {MODELO}, régua {regua_sha}')

feitas = ler_feitas(regua_sha)
pendentes = [(e, i) for e in textos for i in range(1, RODADAS + 1) if (e, i) not in feitas]
print(f'{len(pendentes)} chamadas pendentes')

if args.sem_api:
    e = next(iter(textos))
    print(f'\n--- instrução ({len(regua)} caracteres) ---\n{regua}\n--- texto de {e} ---\n{textos[e]}')
    raise SystemExit

if not pendentes:
    gravar_scores(feitas, todos, regua_sha)
    s = pd.read_csv(ARQ_SCORES)
    print(f'Nada a chamar. {ARQ_SCORES} regravado: {s.gravidade.notna().sum()} de {len(s)} com gravidade; '
          f'distribuição: {s.gravidade.value_counts().sort_index().to_dict()}')
    raise SystemExit

import anthropic
from pydantic import BaseModel


class Score(BaseModel):
    score: Literal[1, 2, 3, 4, 5]


client = anthropic.Anthropic()
os.makedirs(os.path.dirname(ARQ_RESPOSTAS), exist_ok=True)
try:
    with open(ARQ_RESPOSTAS, 'a', encoding='utf-8') as f:
        for k, (e, i) in enumerate(pendentes, 1):
            resp = client.messages.parse(model=MODELO, max_tokens=MAX_TOKENS, system=regua, messages=[{'role': 'user', 'content': textos[e]}], output_format=Score)
            score = resp.parsed_output.score if resp.stop_reason == 'end_turn' and resp.parsed_output else None
            r = {'event_id': e, 'rodada': i, 'modelo': MODELO, 'regua_sha': regua_sha, 'score': score, 'stop_reason': resp.stop_reason,
                 'input_tokens': resp.usage.input_tokens, 'output_tokens': resp.usage.output_tokens, 'request_id': resp._request_id, 'quando_utc': datetime.now(timezone.utc).isoformat()}
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            f.flush()
            feitas[(e, i)] = r
            if k % 10 == 0 or k == len(pendentes):
                print(f'{k}/{len(pendentes)}', flush=True)
except (anthropic.APIStatusError, anthropic.APIConnectionError) as erro:
    print(f'Parou por erro da API: {erro}. O que já respondeu está salvo; rode de novo para continuar.')
finally:
    gravar_scores(feitas, todos, regua_sha)

s = pd.read_csv(ARQ_SCORES)
tokens = [(r['input_tokens'], r['output_tokens']) for r in feitas.values() if r['event_id'] in textos]
print(f'Pronto: {s.gravidade.notna().sum()} de {len(s)} com gravidade; {(s.amplitude > 0).sum()} com notas diferentes entre rodadas; distribuição: {s.gravidade.value_counts().sort_index().to_dict()}')
print(f'Tokens: {sum(t[0] for t in tokens)} de entrada e {sum(t[1] for t in tokens)} de saída')
