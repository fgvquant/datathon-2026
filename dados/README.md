# Bases de dados

Todas as bases ficam aqui, organizadas por tipo. Horários sempre em UTC.

| Pasta | Arquivo | Conteúdo | Situação |
| --- | --- | --- | --- |
| `eventos/` | `coleta_tweets_ataques.xlsx` | primeiro anúncio de cada ataque | 196 ataques, todos com horário (recorte da v19 sem o E519, duplicata do E524; a v19 completa, com 583, está no histórico do Git) |
| `precos/` | `minuto_janela_ataques.csv` | velas de 1 min, ±3h em torno de cada anúncio, 10 moedas | completo para os 196 ataques; versionado como `.csv.gz` |
| `placebo/` | `minuto_janela_placebo.csv.gz` | mesmo formato mais `semanas_antes`: mesma hora, 1 semana antes de cada ataque (2–4 se a janela cruzar outro ataque) | gerado por `scripts/02_baixa_placebo.py` |
| `cci10/` | série diária do CCI10 | pesos do benchmark | com um integrante do grupo |
| `llm/` | `scores.csv` | `event_id`, `gravidade` (1–5, mediana das rodadas), `n_rodadas` | gerado por `scripts/03_score_llm.py` a partir de `respostas.jsonl` (régua em `regua_llm.txt`) |

Ao atualizar uma base, troque o arquivo e mude a versão no nome (ex.: `v19` → `v20`). Não edite à mão uma base
gerada por código.

## eventos/coleta_tweets_ataques.xlsx

Aba `coleta`, uma linha por ataque. A aba `legenda` explica a coleta.

| Coluna | Conteúdo |
| --- | --- |
| `event_id` | identificador do ataque (ex.: E026) |
| `na_lista_bianca` | S/N: está na lista de ataques da Bianca |
| `data_ataque` | data do ataque segundo o DeFiLlama |
| `protocolo`, `ticker` | alvo do ataque |
| `valor_roubado_usd` | valor roubado (DeFiLlama). **Conhecido depois: não usar no LLM nem na regra** |
| `targetType`, `chain` | tipo de alvo e blockchain |
| `perp_simbolo`, `perp_desde` | perpétuo do token na Binance, se houver |
| `busca_x` | link da busca usada no X |
| `responsavel` | quem coletou |
| `fonte`, `handle` | protocolo ou firma de segurança, e o perfil que postou |
| `url`, `tweet_id` | link do post |
| `texto_original` | texto literal do primeiro anúncio (**entrada do LLM**) |
| `menciona_desfecho` | S se o texto já fala em reembolso ou desfecho |
| `obs` | observações da coleta |
| `timestamp_utc` | horário exato do post, em UTC (**define o minuto 0**) |
| `status_coleta` | `verificado no X`, `achado, conferir se é o 1º`, `sem ID (não achado no X)`, `não buscado` |

Regra da coleta: fica o post **mais antigo** entre o protocolo e as firmas de alerta (PeckShieldAlert, CertiKAlert,
BlockSecTeam, SlowMist_Team, CyversAlerts, MistTrack_io, zachxbt).

## precos/minuto_janela_ataques.csv

Formato longo: uma linha por ataque × moeda × minuto.

| Coluna | Conteúdo |
| --- | --- |
| `event_id` | liga com a planilha de eventos |
| `moeda` | uma das 10 moedas do CCI10 |
| `minuto_relativo` | de −180 a +180; 0 é o minuto do anúncio |
| `hora_utc` | horário da vela |
| `hora_referencia_utc` | horário do anúncio |
| `open`, `high`, `low`, `close` | preços da vela |
| `volume`, `quote_volume`, `n_trades`, `taker_buy_base`, `taker_buy_quote` | volumes da vela (Binance) |
| `vela_ausente` | a Binance não tem a vela; preços e volumes ficam vazios (nada é preenchido) |
| `sem_negocio` | vela sem negócios (`n_trades = 0`) |
| `taxa_venda_spot`, `taxa_compra_spot` | corretagem por lado, em fração (0,001 = 0,10%), lida pelo backtest |
| `taxa_total_spot` | corretagem de ida e volta, em fração |
| `tipo_taxa` | `taxa_padrao_assumida`: a taxa vem de `corretagem.taxa_spot_por_lado` no config, igual em todas as velas |

Checagens esperadas: 361 velas por ataque e moeda, nenhuma linha repetida, nenhum preço ≤ 0, máxima ≥ mínima e
minuto 0 batendo com `hora_referencia_utc`.
