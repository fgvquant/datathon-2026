# Datathon 2026 — Drift pós-ataque

Como o primeiro anúncio de um ataque hacker se propaga, minuto a minuto, entre as 10 moedas do CCI10, e se vale
vender para USDT depois da notícia e recomprar quando o efeito passa.

- **Bases de dados:** `dados/` (dicionário em `dados/README.md`)
- **Especificação e regras para o código:** `CLAUDE.md`
- **Documentos do grupo:** `docs/README.md`
- **Resultados:** `results/parametros/` (tabelas e gráficos, versionados)

**Resumo em uma frase:** os ataques geram uma queda pequena, mas mensurável, que se espalha pelas 10 moedas
(−7 a −35 pb em 1 hora, contra quase zero no placebo). Ler a notícia com o LLM não melhora a previsão. E a
estratégia de vender e recomprar só ganha do CCI10 parado por causa de dois ataques; sem eles, perde. A leitura é de
**eficiência de mercado**: o efeito existe, mas é pequeno demais para pagar o custo e a liquidez de quem opera.

---

## 1. A pergunta

Quando um protocolo cripto anuncia que sofreu um ataque (hack, exploit, roubo de fundos), o mercado como um todo
reage? E, se reage, dá para lucrar com isso? Testamos três hipóteses:

| Hipótese | Pergunta | Resposta |
| --- | --- | --- |
| **H1 — Efeito** | O ataque gera uma queda mensurável, e ela se propaga entre as moedas? | **Sim.** Queda de −7 a −35 pb até a vela +60 em todas as moedas; em 6 das 10, fora do IC do placebo. |
| **H2 — Ganho do LLM** | Ler a notícia com um LLM (score de gravidade de 1 a 5) melhora a previsão? | **Não.** O coeficiente do score não é diferente de zero em nenhuma das 3 velas testadas. |
| **H3 — Look-ahead** | O LLM acerta porque reconhece o ataque, e não porque entende a notícia? | Sem resultado neste repositório; como o score não ajuda (H2), não há acerto a explicar. |

## 2. Os dados

| Base | O que é | Onde |
| --- | --- | --- |
| Eventos | 196 ataques de 2020 a 2026 com o horário (UTC, ao segundo) do **primeiro anúncio** no X, conferido à mão. | `dados/eventos/coleta_tweets_ataques.xlsx` |
| Preços | Velas de 1 minuto da Binance spot (par USDT) das 10 moedas do CCI10, de 3 h antes a 3 h depois de cada anúncio, com o volume em USDT e a taxa de corretagem por lado. | `dados/precos/minuto_janela_ataques.csv.gz` |
| Placebo | As mesmas janelas, na mesma hora, 1 semana antes do ataque (2 a 4 semanas se cruzar outro ataque). | `dados/placebo/minuto_janela_placebo.csv.gz` |
| CCI10 | Nível diário e pesos do índice (as 10 maiores moedas por valor de mercado). | `dados/cci10/` |
| Score do LLM | Gravidade de 1 a 5 de cada ataque, lida só do texto do primeiro anúncio. | `dados/llm/scores.csv` |

As 10 moedas: BTC, ETH, BNB, XRP, ADA, TRX, LINK, XLM, BCH e LTC.

**Divisão da amostra (fixada antes de olhar 2026):**
- **Estimação:** 151 ataques de 2020 a 2025. É só daqui que saem os parâmetros do modelo.
- **Negociação (backtest):** 45 ataques de 2026. Nenhum preço de 2026 entra na estimação.

**Convenções.** Horários em UTC. A **vela 0** é o minuto do anúncio (arredondado para baixo); a **vela +1** é o
primeiro minuto inteiro depois da notícia. Se o tweet saiu às 14:03:27, a vela 0 é a das 14:03 e a vela +1 a das
14:04. Retornos são log-retornos de 1 minuto em **pontos-base** (1 pb = 0,01%), calculados dentro de cada janela.

## 3. O modelo: event study intradiário

Seguimos o *event study* clássico (MacKinlay, 1997), mas minuto a minuto.

1. **Retorno normal** de cada moeda em cada ataque: a média do retorno por minuto entre as velas −180 e −61. A
   janela termina 1 hora antes do anúncio porque a queda costuma começar antes da notícia oficial (vazamento,
   alertas de empresas de segurança), e isso contaminaria o "normal".
2. **Retorno anormal** (AR) = retorno observado − retorno normal.
3. **CAR** (retorno anormal acumulado) desde a vela +1: a soma dos AR da vela +1 até a vela h. É essa CAR que interessa
   para operar, porque ninguém consegue vender antes da notícia.
4. **CR(h)** = média da CAR nos 151 ataques de estimação. É a "curva do ataque médio".
5. **Intervalos de confiança** de 90% por **bootstrap por ataque**: 1.000 réplicas, sorteando ataques inteiros com
   reposição (nunca minutos soltos, que são correlacionados entre si).
6. **Placebo:** a mesma conta nas janelas sem ataque. Se o efeito for real, a curva placebo deve ficar perto de zero.

### Resultado (H1)

CR em pb, 151 ataques de 2020–2025, IC de 90% entre colchetes
(`results/parametros/modelo/tabelas/base_tabela.csv`, gráfico em `graficos/base_curvas.png`):

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

Como ler:
- **Há efeito, e ele se propaga.** Todas as moedas caem depois do anúncio, inclusive BTC e ETH, que quase nunca são
  a moeda do protocolo atacado. LINK, ADA e TRX caem mais; XLM quase não cai.
- **Parte da queda vem antes da notícia** (CAR pré de −9 a −31 pb na meia hora antes do anúncio, exceto TRX, contra
  quase zero no placebo): o mercado já reage a sinais que aparecem antes do anúncio oficial (alertas de empresas de
  segurança, movimentações on-chain). Essa parte não dá para operar. A janela é [−30, 0]: com [−10, 0] a queda pré
  era metade (−7 contra −14 pb na média das moedas), e com [−60, 0] o IC passava a cruzar o zero. Ela foi escolhida
  olhando o treino, mas é só descritiva: não entra no backtest.
- **A queda é lenta e pequena.** Leva cerca de 1 hora para chegar a −15 a −35 pb. A **meia-vida** (o minuto em que
  metade da queda posterior à venda já aconteceu) fica perto da vela +36 na maioria das moedas.
- **O placebo fica perto de zero**, e em 6 das 10 moedas (BTC, ETH, TRX, LINK, BCH, LTC) a CR do ataque em +60 fica fora do IC do placebo.
- Em +180 os IC já cruzam o zero: depois de 3 horas, o efeito se perde no ruído.

## 4. O LLM: score de gravidade (H2)

Um LLM (`claude-opus-5-5`) lê **só o texto do primeiro anúncio** e a data, e devolve a gravidade do ataque, de 1 a 5,
segundo uma régua escrita com exemplos (`dados/llm/regua_llm.txt`). O valor roubado, desfechos e notícias
posteriores nunca entram: só o que se sabia no minuto da notícia. Uma rodada por notícia (decisão do grupo).

Distribuição dos 196 scores: 1 → 14, 2 → 91, 3 → 65, 4 → 25, 5 → 1.

**Teste.** Para cada minuto h, regredimos a CAR de cada ataque e moeda no score centrado:
`CAR(h) = a_moeda(h) + b(h) · (g − ḡ) + erro`, com o **mesmo b para as 10 moedas**. Se ler a notícia ajuda, b
deve ser negativo (ataque mais grave → queda maior). Testamos só nas velas +10, +30 e +60, fixadas antes de rodar,
por MQO e, como robustez, por regressão de Huber (menos sensível a ataques extremos).

| Vela | b MQO (pb por ponto de score) | IC 90% | b Huber | IC 90% |
| --- | --- | --- | --- | --- |
| +10 | +3,3 | [−3,7; 11,2] | +0,5 | [−3,1; 3,9] |
| +30 | −2,8 | [−17,4; 11,4] | −0,7 | [−7,5; 6,3] |
| +60 | +5,1 | [−21,8; 35,8] | −1,2 | [−13,7; 11,0] |

**Nenhum b é diferente de zero**, e o sinal muda de uma vela para outra. O score do LLM não ajuda a prever o tamanho
da queda. (`results/parametros/modelo/tabelas/score_comum_teste.csv`)

## 5. A estratégia

### A ideia

Quem tem uma carteira igual ao CCI10 vende as moedas para USDT logo depois do anúncio e recompra quando a queda
termina. Se a queda for maior que o custo de ida e volta, ganha do CCI10 parado.

### O modelo usado para operar: condicional (score + pré-anúncio)

A queda esperada de cada moeda vem de uma regressão do **retorno bruto desde a vela 0** em três preditores, todos
conhecidos no minuto da venda:

```
y_{e,i}(h) = a_i(h) + b(h)·(g_e − ḡ) + c1(h)·(pre60_{e,i} − média) + c2(h)·(pre10_{e,i} − média) + erro
```

- `g_e` = score do LLM do ataque.
- `pre60` = retorno da moeda entre as velas −60 e −11; `pre10` = retorno entre −10 e 0 (o movimento logo antes da
  notícia).
- Inclinações (b, c1, c2) comuns às 10 moedas; um intercepto `a_i(h)` por moeda e por minuto.
- Usamos o retorno **bruto**, sem descontar o retorno normal: com retorno anormal, o preditor e a resposta descontariam
  o mesmo μ estimado, e o erro de μ criaria uma correlação falsa (que aparecia igual no placebo).
- Estimado só nos 151 ataques de 2020–2025.

Coeficientes no drift desde a vela de venda (`results/parametros/modelo/tabelas/condicional_coef.csv`): o único com IC
fora do zero é o `pre60` na vela +30 (+0,14 pb por pb). Ou seja, uma moeda que já vinha caindo na hora antes tende a
continuar caindo um pouco. O score e o `pre10` não são diferentes de zero, e no placebo nenhum coeficiente é.

### A regra

Para cada ataque de 2026 e cada moeda:

1. **Venda** no fechamento da **vela +2** (1 a 2 minutos depois da notícia: tempo de ler e agir).
2. Com a curva esperada do modelo condicional, calcula a **meia-vida**: o primeiro minuto depois da venda em que
   metade da queda esperada já aconteceu, mantida por 3 minutos seguidos. É a vela de **recompra**.
3. **Só vende se** a queda esperada entre a venda e a recompra for maior que o custo de ida e volta.
4. **Prêmio** do ataque:
   `π_e = Σ_{moedas vendidas} peso_CCI10 × parte executada × (1 − P_recompra / P_venda − custos)`.
   O retorno da estratégia é o do CCI10 parado mais π_e.

**Custos e liquidez:**
- Corretagem de **0,10% por lado** (20 pb na ida e volta), lida da base de preços.
- Cada ordem fica limitada a **10% do volume em USDT do minuto** (na vela de venda e na de recompra). Com uma carteira
  de US$ 1 milhão, isso limita muito as moedas pequenas: em média só **7,8%** da posição de cada trade foi executada.
- Pesos do CCI10 do dia anterior ao ataque.

**Comparações:**
- **CCI10 parado** (a linha do zero).
- **"Vender sempre"** (modelo base): a mesma regra com a CR média do treino, sem score nem pré-anúncio.
- **Previsão perfeita:** sabendo todos os preços, recompra no melhor minuto. É o teto do que dá para ganhar.

### Regras contra look-ahead

1. Nada do ataque operado entra na estimação usada para operá-lo (há um `assert` no código).
2. A decisão de venda usa só o texto do anúncio e os preços até a vela de venda. Os custos da decisão são os
   observados na vela de venda.
3. O valor roubado, desfechos e notícias posteriores nunca entram no LLM nem na regra.
4. Todas as variações de parâmetros rodam o backtest inteiro (72 combinações); não escolhemos a melhor variação
   olhando o resultado.

## 6. Resultados do backtest

Combinação principal: venda na vela +2, recompra na meia-vida, treino fixo (2020–2025), carteira de US$ 1 milhão,
corretagem de 0,10% por lado. 45 ataques de 2026.
(`results/parametros/backtest/tabelas/metricas.csv`)

| Estratégia | Ataques operados | Prêmio médio por ataque | Prêmio acumulado | Taxa de acerto | t | Drawdown máx. | Sharpe deflacionado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Condicional | 34 | +1,23 pb | **+55,5 pb** | 15% | 0,87 | 25,8 pb | 0,38 |
| Vender sempre (base) | 45 | −0,02 pb | −1,0 pb | 31% | −0,52 | 2,6 pb | 0,04 |
| Previsão perfeita | 44 | +10,5 pb | +470,7 pb | 100% | 3,10 | 0 | 1,00 |

Taxa de acerto = fração dos ataques operados com prêmio positivo. Sharpe deflacionado (Bailey e López de Prado, 2014)
= probabilidade de o Sharpe verdadeiro superar o melhor esperado por sorte entre as 72 combinações testadas.

**Os trades** (155 vendas em 34 ataques; `results/parametros/backtest/tabelas/trades.csv`):
- Queda esperada média de 28,5 pb; queda realizada média de 11,2 pb. O modelo esperava o dobro da queda que veio.
- O preço caiu entre a venda e a recompra em 50% dos trades. Descontado o custo, só 29% dos trades ganharam.

### Por que o +55 pb não convence

1. **Dois ataques fazem todo o resultado.** O E473 (Saga, +59,6 pb) e o E583 (+21,3 pb) somam mais que o total.
   **Sem os dois, o condicional fica em −25,4 pb**, abaixo do "vender sempre" (−2,4 pb). Só sem o E583 (que a
   especificação já mandava tirar na robustez), fica em +34,1 pb.
2. **O t é 0,87**: o prêmio médio não é diferente de zero.
3. **O modelo condicional foi escolhido depois de ver 2026.** Antes ele era só com o score. O modelo só com score dava
   −8,9 pb na mesma combinação (−10,4 pb sem E473 e E583). O Sharpe deflacionado conta só as combinações do
   condicional, então o 0,38 é otimista.
4. **Varia muito com os parâmetros.** Nas 72 combinações, o prêmio do condicional vai de −49 pb a +157 pb. Com venda
   na vela +10, fica entre −49 e +38 pb; com carteira de US$ 10 milhões, nunca passa de +17 pb (a liquidez não deixa
   executar).
5. **O teto é baixo.** Mesmo com previsão perfeita, o ganho é de +471 pb em 45 ataques (≈ 10 pb por ataque), e é
   isso que um modelo realista precisaria capturar descontando o ruído.

### Conclusão

O efeito do ataque existe (H1) e se espalha pelo mercado, mas é **pequeno (−7 a −35 pb em 1 hora), lento e
ruidoso**. O score do LLM não acrescenta informação (H2). Depois do custo de 20 pb na ida e volta e do limite de
liquidez, nenhuma versão da estratégia ganha do CCI10 parado de forma robusta. É um resultado de **eficiência de
mercado**: a informação do anúncio é incorporada aos preços rápido o bastante, e o que sobra não paga a operação.

## 7. Gráficos

| Arquivo | O que mostra |
| --- | --- |
| `results/parametros/modelo/graficos/base_curvas.png` | CR por moeda, ataque × placebo, com IC |
| `results/parametros/modelo/graficos/condicional_coef.png` | Coeficientes do modelo condicional por minuto, ataque × placebo |
| `results/parametros/backtest/graficos/estrategia_vs_cci10.png` | Estratégia menos CCI10 parado, acumulado em 2026 (pb) |
| `results/parametros/backtest/graficos/estrategia_retorno.png` | Nível da estratégia em 2026 (base 100) |
| `results/parametros/backtest/graficos/modelos_acumulado.png` | Prêmio acumulado: condicional, vender sempre e previsão perfeita |
| `results/parametros/backtest/graficos/trades_esperado_vs_realizado.png` | Esperado × realizado nos 3 melhores e 3 piores trades |
| `results/parametros/backtest/graficos/media_esperado_vs_realizado.png` | Esperado × realizado em média, por moeda, em 2026 |
| `results/parametros/backtest/graficos/cci10_periodo.png` | CCI10 de 2020 a 2026, com os períodos de estimação e negociação |

## 8. Como rodar

```bash
git clone https://github.com/fgvquant/datathon-2026.git
cd datathon-2026
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # e coloque a chave da API do LLM (só para o 03)
```

| Script | O que faz | Precisa rodar? |
| --- | --- | --- |
| `scripts/00_cci10.py` | Monta o CCI10 diário e os pesos a partir dos CSVs do CoinMarketCap | Já está em `dados/cci10/` |
| `scripts/01_baixa_janelas_ataques.py` | Baixa as velas de 1 min da Binance em torno de cada ataque | Já está em `dados/precos/` |
| `scripts/02_baixa_placebo.py` | Baixa as janelas placebo | Já está em `dados/placebo/` |
| `scripts/03_score_llm.py` | Gera o score do LLM (chamadas pagas à API) | Já está em `dados/llm/scores.csv` |
| `scripts/04_modelo.py` | Event study, teste de H2 e modelo condicional, com placebo | Sim (alguns minutos) |
| `scripts/05_backtest.py` | Backtest nas 72 combinações, métricas, trades e gráficos | Sim (~1 min) |

```bash
python scripts/04_modelo.py
python scripts/05_backtest.py
```

Os parâmetros (janelas, velas, custos, carteiras, seeds) estão em `config/parametros.yaml`. As saídas vão para
`results/<nome do config>/`.

No Windows, se der erro de encoding ao redirecionar a saída para um arquivo, rode antes
`set PYTHONIOENCODING=utf-8`.

## 9. Pontos em aberto

- 8 ataques com horário longe da data do DeFiLlama e 3 possíveis não-ataques (E578, E413, E382) continuam na amostra.
- Três pares de ataques a menos de 3 h um do outro (E155–E156, E290–E288, E391–E393) têm janelas sobrepostas.
- A taxa spot da Binance (0,10% por lado) é uma taxa padrão assumida, igual em todo o período.
- A ordem é executada em uma vela só; fatiar em várias velas aumentaria a parte executada nas moedas pequenas.
