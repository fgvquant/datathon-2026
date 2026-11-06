# Bases de dados

Todas as bases ficam aqui, organizadas por tipo. Horários sempre em UTC.

| Pasta | Arquivo | Conteúdo | Situação |
| --- | --- | --- | --- |
| `eventos/` | `coleta_tweets_ataques_v19.xlsx` | primeiro anúncio de cada ataque | 583 ataques, 197 com horário |
| `precos/` | `minuto_janela_ataques.csv` | velas de 1 min, ±3h em torno de cada anúncio, 10 moedas | adicionar; hoje cobre só os ataques antigos |
| `placebo/` | `minuto_janela_placebo.csv` | mesmo formato, uma semana antes de cada ataque, sem ataque | a baixar |
| `cci10/` | série diária do CCI10 | pesos do benchmark | com um integrante do grupo |
| `llm/` | respostas do LLM | uma resposta bruta por notícia e rodada | gerado pelo código |

Ao atualizar uma base, troque o arquivo e mude a versão no nome (ex.: `v19` → `v20`). Não edite à mão uma base
gerada por código.

## eventos/coleta_tweets_ataques_v19.xlsx

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
| `vela_ausente` | 1 se não houve negociação no minuto |

Checagens esperadas: 361 velas por ataque e moeda, nenhuma linha repetida, nenhum preço ≤ 0, máxima ≥ mínima e
minuto 0 batendo com `hora_referencia_utc`.
