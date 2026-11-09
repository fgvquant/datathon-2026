# Briefing para montar a apresentação — Datathon 2026, drift pós-ataque

Este arquivo é para o Claude que vai ajudar a montar os slides. Ele resume o projeto, as decisões que o grupo tomou
(com data e motivo), como o código ficou e quais números usar. Tudo aqui foi conferido contra `results/parametros/`
em 08/11/2026. **Use só estes números ou os arquivos de `results/`; não invente nem arredonde para cima.**

- Entrega: **domingo, 08/11/2026, até 23:59** (backtest e slides).
- Idioma: português.
- Fontes mais detalhadas: `README.md` (resumo completo da estratégia), `CLAUDE.md` (especificação e regras),
  `docs/explicacao-do-modelo.md` (explicação didática do modelo).

---

## 1. A história em 30 segundos

Quando um protocolo cripto anuncia que foi hackeado, o mercado inteiro cai um pouco, inclusive BTC e ETH. Medimos essa
queda minuto a minuto nas 10 moedas do CCI10 e testamos se dá para lucrar vendendo para USDT logo depois do anúncio e
recomprando quando a queda passa.

**Mensagem central:** o efeito existe, mas é pequeno, lento e ruidoso. Ler a notícia com um LLM não ajuda a prever o
tamanho da queda. Depois do custo e do limite de liquidez, nenhuma versão da estratégia ganha do CCI10 parado de forma
robusta. A conclusão é de **eficiência de mercado**, que o grupo combinou desde o início que também seria um resultado
válido.

## 2. Hipóteses e respostas

| Hipótese | Pergunta | Resposta |
| --- | --- | --- |
| H1 — Efeito | O ataque gera queda mensurável, e ela se propaga entre as moedas? | **Sim.** −7 a −35 pb até a vela +60 em todas as moedas; em 6 das 10 (BTC, ETH, TRX, LINK, BCH, LTC), fora do IC do placebo. |
| H2 — Ganho do LLM | O score de gravidade do LLM melhora a previsão? | **Não.** O b comum não é diferente de zero em nenhuma das 3 velas testadas (+10, +30, +60). |
| H3 — Look-ahead do LLM | O LLM acerta porque reconhece o ataque? | Sem resultado no repositório. Como o score não ajuda (H2), não há acerto a explicar. |

## 3. Dados

- **196 ataques** de 2020 a 2026, com o horário (UTC, ao segundo) do **primeiro anúncio** no X, conferido à mão
  (`dados/eventos/coleta_tweets_ataques.xlsx`). O E519 saiu da base por ser duplicata do E524 (mesmo incidente).
- **Preços:** velas de 1 minuto da Binance spot (pares USDT) das 10 moedas (BTC, ETH, BNB, XRP, ADA, TRX, LINK, XLM, BCH,
  LTC), de 3 h antes a 3 h depois de cada anúncio (361 velas por ataque e moeda; 707.560 linhas; nenhuma vela ausente).
- **Placebo:** a mesma janela na mesma hora, 1 semana antes (2 a 4 semanas se cruzar outro ataque).
- **CCI10:** nível diário e pesos (benchmark).
- **Score do LLM:** gravidade de 1 a 5 de cada ataque. Distribuição: 1 → 14, 2 → 91, 3 → 65, 4 → 25, 5 → 1.
- **Divisão:** estimação = **151 ataques de 2020–2025**; negociação (backtest) = **45 ataques de 2026**. Nenhum preço
  de 2026 entra na estimação.

Convenções para os slides: vela 0 = minuto do anúncio (arredondado para baixo); vela +1 = primeiro minuto inteiro
depois da notícia. Retornos em **pontos-base** (1 pb = 0,01%).

## 4. Linha do tempo das decisões

| Data | Decisão | Motivo |
| --- | --- | --- |
| 06/11 | H2 testada pelo coeficiente b do score na regressão da CAR, e no backtest contra o "vender sempre". | Comparação simples e direta do ganho do LLM. |
| 07/11 | **Modelo principal = event study com CAR** (MacKinlay, 1997). O VAR-X com faixas virou só robustez. | As faixas do VAR eram escolha nossa, feita olhando a média, e os coeficientes eram fracos (~0,12): a propagação pelo VAR quase não mudava a curva. |
| 07/11 | **Retorno normal = velas −180 a −61** (termina 1 h antes do anúncio). | A queda começa antes da notícia (vazamento). Com [−180, −11], o μ absorvia ~8 pb da queda. |
| 07/11 | **Placebo A:** mesma hora, 1 semana antes (2–4 semanas se cruzar outro ataque). | Controla hora do dia e dia da semana. |
| 07/11 | **LLM com 1 rodada por notícia** (a especificação pedia 3 a 5 e mediana). | Decisão do grupo, por custo e prazo. |
| 07/11 | **Treino 2020–2025, negociação em 2026.** | Teste fora da amostra, sem nenhum preço de 2026 na estimação. |
| 08/11 | **Recompra só na meia-vida** (o vale ficou só descritivo). | O vale é instável (IC de ~30 a ~170 minutos); a meia-vida é mais estável. |
| 08/11 | **H2 com b comum às 10 moedas**, testado só em +10, +30 e +60, IC de 90% sem Bonferroni; Huber como robustez. | Menos testes e mais potência que 80 coeficientes por moeda. |
| 08/11 | **Corretagem de 0,10% por lado** (taxa padrão Binance VIP 0), gravada pelo 01 a partir do config. | Com BNB seria 0,075%; 0,10% é conservador. |
| 08/11 | **Modelo condicional** (score + movimento pré-anúncio) virou a estratégia principal do backtest, no lugar do modelo só com score. | Ver o aviso logo abaixo. |
| 08/11 | **CAR pré em [−30, 0]** (era [−10, 0]). | Nos 151 de treino, a queda pré dobra (−7 → −14 pb na média das moedas) com o placebo perto de zero; [−60, 0] só somava ruído. É só descritiva. |
| 08/11 | Limpeza: saíram as curvas CR(·; g), o vale e a meia-vida por score, os 80 b por moeda e a suíte de testes. | Não eram mais usados; o código ficou só com o que alimenta os resultados. |

**Aviso importante sobre o modelo condicional.** Ele foi adotado como estratégia principal *depois* de o grupo ver o
resultado de 2026 com o modelo só com score. O modelo só com score dava **−8,9 pb** na combinação principal
(−10,4 pb sem E473 e E583). O Sharpe deflacionado (0,38) conta só as combinações do condicional, então é otimista.
O README atual não menciona isso; o comentário em `config/parametros.yaml` (seção `condicional`) ainda menciona.
**Não apresente o +55 pb como resultado fora da amostra limpo.** Uma forma honesta e leve: "a versão pré-registrada
(só score) dá −8,9 pb; a extensão com o movimento pré-anúncio dá +55 pb, mas é exploratória e depende de dois
ataques." Confirme com o usuário como ele quer tratar isso nos slides.

## 5. O modelo (o que explicar no slide de metodologia)

1. **Retorno normal** μ de cada moeda em cada ataque = média do retorno por minuto entre as velas −180 e −61.
2. **Retorno anormal** AR = retorno − μ.
3. **CAR** desde a vela +1 = soma dos AR da vela +1 até h (é o que dá para operar; ninguém vende antes da notícia).
4. **CR(h)** = média da CAR nos 151 ataques de treino ("curva do ataque médio").
5. **IC de 90%** por **bootstrap por ataque** (1.000 réplicas, sorteando ataques inteiros).
6. **Placebo:** a mesma conta nas janelas sem ataque.
7. **CAR pré** = soma dos AR de −30 a 0. É só descritiva: mede o vazamento antes do anúncio e não entra em nenhuma
   estimação ou decisão.
8. **Meia-vida** = primeiro minuto depois da venda em que metade da queda posterior já aconteceu, mantida por 3 minutos
   seguidos.

### Resultados do modelo base (H1), 151 ataques de treino, pb, IC de 90%

| Moeda | CAR pré (−30 a 0) | CR +10 | CR +30 | CR +60 | CR +60 placebo | Meia-vida (vela) |
| --- | --- | --- | --- | --- | --- | --- |
| BTC | −8,6 [−15,9; −1,1] | −4,5 [−8,2; −1,0] | −10,7 [−20,4; −2,0] | −14,9 [−25,0; −5,3] | −4,6 [−13,4; 3,9] | 35 |
| ETH | −11,3 [−21,9; −1,2] | −6,6 [−11,7; −2,3] | −12,4 [−23,6; −2,4] | −23,8 [−41,2; −8,7] | −3,3 [−14,7; 8,4] | 36 |
| BNB | −8,9 [−19,4; 1,2] | −6,1 [−13,3; 0,2] | −10,8 [−23,3; 1,3] | −21,5 [−43,5; −2,7] | −10,2 [−22,3; 1,4] | 39 |
| XRP | −14,2 [−25,6; −3,2] | −11,5 [−19,6; −3,8] | −14,8 [−30,0; −0,7] | −22,1 [−47,2; 0,3] | −10,6 [−28,8; 7,2] | 36 |
| ADA | −30,5 [−43,6; −17,5] | −8,8 [−15,9; −2,3] | −20,2 [−34,4; −6,3] | −29,3 [−52,5; −6,6] | −11,7 [−29,4; 5,4] | 36 |
| TRX | −1,0 [−12,7; 14,1] | −6,7 [−12,2; −1,7] | −15,9 [−27,1; −5,9] | −27,5 [−42,9; −13,6] | 2,7 [−7,1; 12,7] | 27 |
| LINK | −18,6 [−30,1; −7,4] | −13,6 [−20,8; −6,4] | −23,0 [−36,9; −10,1] | −35,3 [−56,0; −14,8] | −0,4 [−16,5; 16,1] | 36 |
| XLM | −13,0 [−23,9; −2,4] | −5,1 [−11,9; 1,6] | −9,3 [−23,1; 4,0] | −6,9 [−29,1; 14,4] | −6,8 [−23,1; 9,7] | 12 |
| BCH | −20,9 [−32,1; −9,7] | −9,9 [−17,2; −3,4] | −12,3 [−24,5; −0,7] | −14,6 [−34,5; 2,5] | 3,6 [−11,4; 20,2] | 17 |
| LTC | −16,2 [−27,1; −5,9] | −8,6 [−14,9; −2,8] | −10,0 [−21,7; 1,3] | −17,5 [−36,7; 0,2] | −1,1 [−16,1; 13,7] | 37 |

Mensagens para o slide:
- Todas as moedas caem, inclusive BTC e ETH, que quase nunca são a moeda atacada: **o efeito se propaga**.
- **Parte da queda vem antes do anúncio** (CAR pré de −9 a −31 pb; 8 das 10 moedas com IC fora do zero): o mercado
  reage a sinais anteriores (alertas de empresas de segurança, movimentações on-chain).
- A queda é **lenta**: leva ~1 hora para chegar a −15 a −35 pb; meia-vida perto da vela +36.
- Em +180 todos os IC cruzam o zero: depois de 3 horas o efeito se perde no ruído.

### H2: score do LLM (b comum às moedas, pb por ponto de score)

| Vela | b MQO | IC 90% | b Huber | IC 90% |
| --- | --- | --- | --- | --- |
| +10 | +3,3 | [−3,7; 11,2] | +0,5 | [−3,1; 3,9] |
| +30 | −2,8 | [−17,4; 11,4] | −0,7 | [−7,5; 6,3] |
| +60 | +5,1 | [−21,8; 35,8] | −1,2 | [−13,7; 11,0] |

Nenhum b é diferente de zero e o sinal troca entre as velas: o score não explica o tamanho da queda. São 3 testes,
sem correção para múltiplos testes (dizer isso no slide).

O LLM recebe só o texto do primeiro anúncio e a data; nunca o valor roubado, desfechos ou notícias posteriores.
Modelo `claude-opus-5-5`, régua com exemplos em `dados/llm/regua_llm.txt`.

## 6. A estratégia e o backtest

**Modelo condicional** (o usado para operar): regressão do retorno bruto desde a vela 0 em três preditores conhecidos
na hora da venda, com inclinações comuns às moedas e intercepto por moeda e minuto, estimada nos 151 de treino:
- score do LLM;
- `pre60` = retorno entre as velas −60 e −11;
- `pre10` = retorno entre as velas −10 e 0.

Retorno bruto, não anormal: com AR, preditor e resposta descontariam o mesmo μ, e o erro de μ criaria uma correlação
falsa (aparecia igual no placebo). Único coeficiente com IC fora do zero: `pre60` na vela +30 (+0,14 pb por pb), ou seja,
moeda que já vinha caindo continua caindo um pouco. No placebo, nenhum coeficiente é diferente de zero.

**Regra**, para cada ataque de 2026 e cada moeda:
1. Venda no fechamento da **vela +2**.
2. Recompra na **meia-vida** da curva esperada.
3. Só vende se a queda esperada entre a venda e a recompra supera o custo de ida e volta (20 pb).
4. Prêmio do ataque: `π_e = Σ peso_CCI10 × parte executada × (1 − P_recompra/P_venda − custos)`.
5. Ordem limitada a **10% do volume em USDT do minuto** (venda e recompra). Pesos do CCI10 do dia anterior.

**Comparações:** CCI10 parado; "vender sempre" (CR média do treino, sem preditores); previsão perfeita (teto).

**Regras contra look-ahead:** o ataque operado nunca entra no próprio treino (`assert` no código); a decisão usa só o
texto e os preços até a vela de venda; o valor roubado nunca entra; todas as 72 combinações rodam o backtest inteiro.

### Resultado principal

Venda +2, recompra na meia-vida, treino fixo, US$ 1 milhão, corretagem 0,10%/lado, 45 ataques de 2026:

| Estratégia | Ataques operados | Prêmio médio | Prêmio acumulado | Taxa de acerto | t | Drawdown máx. | Sharpe deflacionado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Condicional | 34 | +1,23 pb | **+55,5 pb** | 15% | 0,87 | 25,8 pb | 0,38 |
| Vender sempre | 45 | −0,02 pb | −1,0 pb | 31% | −0,52 | 2,6 pb | 0,04 |
| Previsão perfeita | 44 | +10,5 pb | +470,7 pb | 100% | 3,10 | 0 | 1,00 |

- O CCI10 caiu em 2026: nível 85,52 (base 100 em 31/12/2025) no último ataque (29/10/2026); a estratégia, 85,99.
- **155 trades** em 34 ataques. Queda esperada média de 28,5 pb; realizada de 11,2 pb (o modelo esperava o dobro).
  O preço caiu em 50% dos trades; descontado o custo, 29% ganharam. Em média só **7,8%** da posição foi executada
  (liquidez).

### Por que o +55 pb não se sustenta (slide de robustez)

1. **Dois ataques fazem tudo:** E473 (Saga, +59,6 pb) e E583 (hack da própria Binance, +21,3 pb). Sem os dois, o
   condicional dá **−25,4 pb**, abaixo do "vender sempre" (−2,4 pb). Só sem o E583 (que a especificação mandava tirar
   na robustez), +34,1 pb.
2. **t = 0,87**: o prêmio médio não é diferente de zero.
3. **Escolha do modelo depois de ver 2026** (seção 4): o modelo só com score dava −8,9 pb.
4. **Instável nos parâmetros:** nas 72 combinações, de −49 a +157 pb. Com venda na vela +10, de −49 a +38 pb; com
   US$ 10 milhões, nunca passa de +17 pb.
5. **Teto baixo:** mesmo com previsão perfeita, ≈ 10 pb por ataque.

## 7. Gráficos prontos (usar nos slides)

| Arquivo | Slide sugerido |
| --- | --- |
| `results/parametros/backtest/graficos/cci10_periodo.png` | Dados: CCI10 2020–2026 com estimação × negociação e os ataques |
| `results/parametros/modelo/graficos/base_curvas.png` | H1: CR por moeda, ataque × placebo, com IC |
| `results/parametros/modelo/graficos/condicional_coef.png` | Modelo condicional: coeficientes por minuto, ataque × placebo |
| `results/parametros/backtest/graficos/modelos_acumulado.png` | Backtest: condicional × vender sempre × previsão perfeita |
| `results/parametros/backtest/graficos/estrategia_vs_cci10.png` | Backtest: estratégia − CCI10 acumulado (pb) |
| `results/parametros/backtest/graficos/estrategia_retorno.png` | Backtest: nível da estratégia em 2026 |
| `results/parametros/backtest/graficos/media_esperado_vs_realizado.png` | Por que não funciona: esperado × realizado por moeda |
| `results/parametros/backtest/graficos/trades_esperado_vs_realizado.png` | Exemplos: 3 melhores e 3 piores trades |

Tabela interativa dos 155 trades (para mostrar ao vivo ou tirar print):
https://claude.ai/artifact/JB4bPuPcmB2bj9G5NVNrZe

## 8. Como o código ficou

```
config/parametros.yaml        todos os parâmetros (janelas, velas, custos, carteiras, seeds)
src/modelo.py                 CAR, bootstrap, regressões (score, b comum, Huber, condicional), vale e meia-vida
src/backtest.py               regra de venda, previsão perfeita, prêmio com teto de liquidez, métricas, Sharpe deflacionado
src/placebo.py, src/cci10.py  janelas placebo e índice CCI10
scripts/00_cci10.py           CCI10 diário e pesos (CoinMarketCap)
scripts/01_baixa_janelas_ataques.py   velas de 1 min da Binance + corretagem do config em cada vela
scripts/02_baixa_placebo.py   janelas placebo
scripts/03_score_llm.py       score do LLM (único que chama a API paga)
scripts/04_modelo.py          event study, H2 e modelo condicional, com placebo → results/parametros/modelo/
scripts/05_backtest.py        72 combinações, métricas, trades e gráficos → results/parametros/backtest/
results/parametros/           saídas versionadas no Git
```

- Pipeline reproduzível do zero: 01 → 02 → (03) → 04 → 05. O 03 precisa de chave de API, mas o `scores.csv` já está
  versionado.
- Grade do backtest: venda em +1, +2, +5, +10 × treino fixo ou expansivo × carteira US$ 100 mil, 1 mi, 10 mi ×
  custo (corretagem, Binance com spread e impacto, sem custo) = 72 combinações.
- Seeds fixas em tudo que é aleatório. Não há suíte de testes (removida a pedido); as travas de look-ahead ficam como
  `assert` no código.
- Tudo está na `main` do GitHub (https://github.com/fgvquant/datathon-2026).

## 9. Roteiro sugerido de slides (~12)

1. Título e pergunta.
2. Motivação: um hack num protocolo pequeno derruba BTC e ETH? Dá para lucrar?
3. Hipóteses H1, H2, H3.
4. Dados: 196 primeiros anúncios conferidos à mão, velas de 1 min, placebo, divisão treino/2026 (`cci10_periodo.png`).
5. Metodologia: event study minuto a minuto, retorno normal [−180, −61], bootstrap por ataque.
6. H1: curvas CR × placebo (`base_curvas.png`) e a tabela resumida (CR +60 e meia-vida).
7. Vazamento: CAR pré de −9 a −31 pb na meia hora antes do anúncio.
8. H2: o score do LLM não ajuda (tabela do b comum).
9. Estratégia: regra de venda, meia-vida, custo, liquidez, travas contra look-ahead.
10. Backtest: tabela principal e `modelos_acumulado.png`.
11. Robustez: dois ataques, t = 0,87, instabilidade nos parâmetros, escolha do modelo (`media_esperado_vs_realizado.png`).
12. Conclusão: eficiência de mercado; limitações e próximos passos.

**Limitações para o último slide:** 8 ataques com horário longe da data do DeFiLlama e 3 possíveis não-ataques
(E578, E413, E382) seguem na amostra; três pares de ataques com janelas sobrepostas (E155–E156, E290–E288, E391–E393);
LLM com 1 rodada; taxa de corretagem assumida constante; ordem executada em uma vela só.

**Próximos passos possíveis:** fatiar a ordem em várias velas; rodar H3 (nome do protocolo mascarado, ataques após o
corte de conhecimento do modelo); validar a escolha do modelo em 2025 antes de olhar 2026; operar perpétuos (venda a
descoberto) em vez de só vender a carteira.
