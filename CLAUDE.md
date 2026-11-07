# CLAUDE.md — Datathon 2026: drift pós-ataque

Guia para o Claude (e para o grupo) escrever o código deste projeto. Leia inteiro antes de começar qualquer tarefa.
Responda e comente o código em **português**.

## 1. O projeto em um parágrafo

Medimos como o **primeiro anúncio** de um ataque hacker se propaga, minuto a minuto, entre as 10 moedas do CCI10,
e testamos se vale vender a moeda para USDT logo depois da notícia e recomprar quando o efeito passa.
A ordem do trabalho é **modelo → LLM → backtest**. Se o efeito não aparecer no modelo, o LLM e o backtest viram a base
de um argumento de eficiência de mercado, que também é um resultado válido.

- Entrega: **domingo, 08/11/2026, até 23:59** (backtest e slides).
- Especificação completa: documento "Drift pós-ataque — Formalização do modelo" (link em `docs/README.md`).
  Se este arquivo e o documento divergirem, pergunte ao grupo antes de escolher.

| Hipótese | Pergunta | Onde é testada |
| --- | --- | --- |
| H1 — Efeito | o ataque gera queda mensurável e ela se propaga entre as moedas? | modelo, contra placebo |
| H2 — Ganho do LLM | ler a notícia com o LLM (score de gravidade) melhora o modelo e o backtest em relação ao modelo base? | b(h) na regressão da CAR no score e backtest |
| H3 — Look-ahead | o LLM acerta porque reconhece o ataque, e não porque entende a notícia? | nome do protocolo mascarado; ataques após o corte de conhecimento do modelo |

## 2. Estrutura do repositório

```
dados/
  eventos/    coleta_tweets_ataques.xlsx — primeiro anúncio de cada ataque (os 197 com horário da v19)
  precos/     minuto_janela_ataques.csv — velas de 1 min, ±3h em torno de cada anúncio
  placebo/    janelas sem ataque (mesmo horário, uma semana antes)
  cci10/      série diária e pesos do CCI10
  llm/        scores.csv — score de gravidade do LLM por ataque (gerado fora deste repositório)
config/       parâmetros em YAML (janela do retorno normal, latência, custos...). Nada disso fica fixo no código.
src/          funções reutilizáveis (pacote Python)
scripts/      scripts curtos que chamam src/ e salvam em results/
notebooks/    exploração e gráficos; nenhuma lógica que outro arquivo precise
tests/        pytest
results/      saídas (fora do Git)
docs/         links e notas
```

O dicionário das bases está em `dados/README.md`.

## 3. Dados: regras que valem em todo o código

- **Horários sempre em UTC.**
- **Minuto 0** = minuto do `timestamp_utc` do primeiro anúncio (planilha de eventos), arredondado para baixo.
- **Vela** = candle de 1 minuto. A **vela +1** é a primeira vela inteira depois da notícia. Se o tweet saiu às
  14:03:27, a vela 0 é a das 14:03 e a vela +1 é a das 14:04.
- **Retorno** = log-retorno de 1 minuto em pontos-base, `1e4 * (ln P_t - ln P_{t-1})`, calculado **dentro** de cada
  janela. Nunca calcule retorno entre a última vela de um ataque e a primeira do seguinte.
- **E519 e E524 são o mesmo incidente** (horários a 2 s de distância): use só um.
- **Amostra principal:** ataques com `timestamp_utc` preenchido. Na v19 são 197 de 583 (188 "verificado no X" e
  9 "achado, conferir se é o 1º"); 122 até 2024 e 75 em 2025–26.
- **Treino e negociação (decidido pelo grupo):** a negociação são os **45 ataques de 2026**; o modelo é estimado só
  com os **151 de 2020–2025** (`amostra.inicio_teste` no config). Nenhum preço de 2026 entra na estimação.
- **Saem só nos testes de robustez:** E583, rug pulls e anúncios não confirmados.
- **Pontos em aberto na base v19** (não decida sozinho; sinalize ao grupo):
  - 8 ataques com horário a mais de 2 dias antes ou 3 dias depois da data do DeFiLlama:
    E103, E089, E141, E347, E257, E010, E259, E519.
  - Possíveis não-ataques entre os "conferir": E578 (CoinDCX nega o hack), E413 (Bitget, negociação anormal),
    E382 (post da Arkham).
  - 4 horários com segundos zerados (ex.: E257 às 20:20:00), que parecem digitados à mão.
  - 20 primeiros anúncios já mencionam reembolso ou desfecho (`menciona_desfecho = S`).
- `minuto_janela_ataques.csv` cobre os 197 ataques da amostra principal (v19). Faltam as janelas placebo (Binance
  spot, velas de 1 min, `data.binance.vision`), **no mesmo formato**.

## 4. A. Modelo: event study intradiário (CAR)

Decidido em 07/11: o modelo principal é um **event study** (MacKinlay, 1997) com o retorno anormal acumulado (CAR)
de cada ataque, minuto a minuto. O VAR-X com dummies por faixa passa a ser só robustez (fim desta seção). Motivo:
as faixas eram uma escolha nossa, feita olhando a média dos ataques, e os coeficientes A do VAR eram fracos (~0,12),
então a propagação pelo VAR quase não mudava a curva. A comparação completa está no documento do grupo
"VAR-X com faixas × event study (CAR)".

Notação: e = ataque, i = moeda, t = minuto relativo ao anúncio (vela). h = índice a partir da vela +1 (h = vela − 1).

```
Retorno normal:     μ_{e,i} = média de r_{e,i,t} para t em [−180, −11]       (pb por minuto, da própria janela)
Retorno anormal:    AR_{e,i,t} = r_{e,i,t} − μ_{e,i}
CAR desde +1:       CAR_{e,i}(h) = Σ_{vela 1..h+1} AR_{e,i,t}                 h = 0..179  ← é este que importa para operar
CAR pré:            CAR_pre_{e,i} = Σ_{t = −10..0} AR_{e,i,t}                 descritivo; não dá para operar

Modelo base (H1):         CR_i(h) = média de CAR_{e,i}(h) nos ataques da estimação
Modelo com score (H2):    CAR_{e,i}(h) = a_i(h) + b_i(h) (g_e − ḡ) + ε_{e,i}(h)    MQO por moeda e por h
                          CR_i(h; g) = a_i(h) + b_i(h) (g − ḡ)
```

- **Janela do retorno normal** ([−180, −11]) fica em `config/`. Ela não inclui a faixa pré (−10 a −1) para o
  vazamento antes do anúncio não contaminar o "normal".
- **Uma CAR por minuto, sem faixas:** a curva tem a forma que os dados mostrarem, até +180. Nada é imposto depois de +60.
- `g_e` = score de gravidade (1 a 5, seção 6). `ḡ` = média dos scores dos ataques usados na estimação; centrar mantém
  `a_i(h)` igual ao efeito do ataque médio, comparável ao modelo base. No modelo base, `b = 0` e `a_i(h) = CR_i(h)`.
- O score só existe a partir do anúncio; ele explica a CAR desde a vela +1 (e, descritivamente, a CAR da vela 0).
  Não use o score para explicar a CAR pré.
- **Janelas sobrepostas:** três pares de ataques estão a menos de 3 h um do outro (E155–E156, E290–E288, E391–E393).
  O retorno normal e a CAR de um deles contêm o outro ataque. Sinalize no relatório e rode a robustez sem esses pares.
- **Intervalos:** bootstrap **por ataque** (reamostra ataques inteiros com reposição, 1.000 réplicas), nunca por
  minuto. IC percentil de 90%. No modelo com score, o score viaja com o ataque reamostrado e `ḡ` é recalculado em
  cada réplica.
- **Placebo:** a mesma conta nas janelas placebo (μ da própria janela placebo); o efeito deve ser nulo.
- **Propagação entre as moedas (H1, descritiva):** comparar a CR por moeda (quanto cai) e o minuto da meia-vida por
  moeda (quando cai). Robustez opcional: regressão do AR de cada moeda no AR defasado do BTC.
- **Entrega do modelo base:** tabela por moeda com CAR_pre, CR nas velas **+1, +2, +5, +10, +15, +30, +60 e +180**,
  minuto do vale e meia-vida, cada um com IC, ao lado do placebo; e um gráfico da curva CR completa por moeda, com IC.
- **Entrega do modelo com score:** `b_i(h)` por moeda nas mesmas velas, com IC; curvas `CR(·; g)` para g = 1..5;
  vale e meia-vida por score.

**Robustez (não é o modelo principal):**
- **Modelo de mercado:** `AR = r − (α + β r_m)`, com α e β estimados em [−180, −11]. Cuidado: se o mercado inteiro
  cai com o ataque, descontar o mercado tira parte do efeito; leia junto com o modelo de média constante.
- **VAR-X empilhado com dummies por faixa** (especificação anterior):
  `y_{e,t} = α_e + Σ_{j=1..p} A_j y_{e,t-j} + Σ_f [b0^f + b1^f (g_e − ḡ)] D^f_{e,t} + u_{e,t}`, faixas
  −10 a −1, 0, +1, +2 a +5, +6 a +15, +16 a +60; efeito fixo por janela (within), p por BIC na mesma amostra,
  defasagens sem cruzar janelas, Cholesky BTC, ETH e demais por valor de mercado. Se a curva CR do VAR-X ficar
  parecida com a do event study, isso reforça o resultado.

## 5. Prazo de saída: vale e meia-vida

Com a venda na vela h_e (base: vela +2, ou seja, índice h_e = 1), sobre a curva `CR(·; g)` de cada moeda:

```
h_vale = argmin_{h ≥ h_e} CR(h; g)
h_1/2  = min{ h ≥ h_e : CR(h; g) − CR(h_vale; g) ≤ 0,5 [CR(h_e; g) − CR(h_vale; g)] }
```

- Nos dois casos, a condição precisa valer por **3 minutos seguidos**.
- Cuidado com a conversão entre **vela** (+1, +2, ...) e **índice h** (0, 1, ...): `h = vela − 1`.
- O vale é instável quando o fundo é plano e cai no fim da janela (+180) se não há reversão. A meia-vida captura
  metade da queda, mas é mais estável. O backtest testa as duas.

## 6. B. LLM: score de gravidade

- O LLM devolve **só a gravidade** do ataque: inteiro de 1 a 5, lido do texto do primeiro anúncio, segundo uma régua
  escrita com exemplos. Não há `confirma_ataque`, `moedas_expostas` nem regressão de calibração.
- O score entra na regressão em corte transversal da CAR (seção 4).
- Quem gera os scores é outro integrante do grupo. Este repositório só lê `dados/llm/scores.csv`, com as colunas
  `event_id`, `gravidade` (mediana das rodadas, inteiro 1–5) e `n_rodadas`. Nenhum código aqui chama a API.
- Enquanto `scores.csv` não existir, o modelo com score e o backtest rodam com scores **SIMULADOS** (seed fixa no
  config), e toda saída é marcada como tal (nome do arquivo e título).
- Regras da geração (para quem roda o LLM): entrada só `texto_original` e a data, **nunca** `valor_roubado_usd`;
  prompt e régua versionados e congelados antes do teste; 3 a 5 rodadas por notícia e mediana; régua validada com
  30 a 40 notícias rotuladas à mão; H3 com o nome do protocolo mascarado e com os ataques após o corte de
  conhecimento do modelo.

## 7. C. Backtest

- **Período de negociação:** os 45 ataques de 2026 (`periodo == "teste"`). A estimação de a, b e ḡ (logo, das
  curvas CR, do vale e da meia-vida) usa os 151 ataques de 2020–2025. Se o backtest reestimar ao longo de 2026
  (janela expansiva), um ataque de 2026 só entra no treino **depois** de operado, nunca no dele mesmo.
- **Fica fixo** ao longo do backtest: scores do LLM (prompt e régua), janela do retorno normal, vela do ataque,
  regra de saída e custos.
- **Regra:** com `CR(·; g_e)` do treino, vender a moeda i se
  `−[CR_i(h_recompra; g_e) − CR_i(h_venda; g_e)] > custo de ida e volta`.
- **Venda** na vela +2 (testar +1, +5, +10). **Recompra** no vale ou na meia-vida da moeda (config).
- **Prêmio** de cada ataque e retorno da estratégia (pesos diários de `dados/cci10/cci10_pesos_diarios.csv`):

```
π_e = Σ_{i vendidas} w_i^{CCI10} (1 − P_{i,recompra} / P_{i,venda}) − custos_e
R_e(estratégia) = R_e(CCI10) + π_e
```

- **Custos** por lado no config: taxa de ~0,10% (confirmar na Binance), spread e impacto; ordem limitada a uma fração
  (5–10%) do volume do minuto; carteiras de US$ 100 mil, 1 mi e 10 mi.
- **Comparações:** CCI10 parado; "vender sempre" = a mesma regra com o modelo base (`b = 0`); datas placebo;
  previsão perfeita. **H2** = (a) `b ≠ 0` por bootstrap e (b) modelo com score contra "vender sempre".
- **Métricas:** prêmio por operação e acumulado, taxa de acerto, Information Ratio, drawdown máximo, teste t e
  Sharpe deflacionado (Bailey e López de Prado, 2014).

## 8. Regras invioláveis (look-ahead)

1. Nada sobre o ataque que está sendo operado entra na estimação usada para operá-lo.
2. A decisão de venda usa só informação disponível até a vela de venda: o texto do anúncio e os preços até ali.
3. `valor_roubado_usd`, desfechos e notícias posteriores nunca entram no LLM nem na regra.
4. Placebo, robustez e variações de parâmetros rodam o backtest **inteiro** de novo; nunca escolha a melhor variação
   olhando o resultado do teste.
5. Se um resultado parecer bom demais, procure look-ahead antes de comemorar.

## 9. Convenções de código

- Python 3.11, `pandas`, `numpy`, `statsmodels`, `pyyaml`, `pytest`; `anthropic` para o LLM.
- Funções em `src/`, com docstring dizendo o que entra, o que sai e em que unidade (pb, %, minutos, velas).
- Scripts em `scripts/` curtos, numerados na ordem do pipeline (`01_...`, `02_...`), lendo parâmetros de `config/`
  e salvando em `results/` com o nome da configuração usada.
- Nomes em português, como no documento (`queda`, `vela`, `vale`, `meia_vida`, `premio`).
- Seeds fixas em tudo que é aleatório (bootstrap, placebo, sorteios).
- **Testes com dados simulados** em que a resposta é conhecida: por exemplo, plantar um ataque de −60 pb no BTC na
  vela +1 e −5 pb/min de +2 a +10 e conferir que a CR recupera −105 pb em +10; plantar `b = −4` e conferir a
  estimativa; conferir que o retorno normal usa só [−180, −11]; conferir que o backtest nunca vê ataques futuros.
- Fluxo Git: uma branch por tarefa (`modelo/`, `llm/`, `backtest/`, `dados/`), Pull Request revisado por outra
  pessoa do grupo, testes verdes antes do merge.
- **Mensagens de commit (obrigatório)** seguem o [gitmoji](https://gitmoji.dev), no formato
  `<emoji> comando: nome-do-commit`, com o emoji em código (`:sparkles:`), em minúsculas e em português. Exemplos:
  - `:sparkles: feat: calcula-car-por-ataque`
  - `:bug: fix: retorno-nao-cruza-janelas`
  - `:white_check_mark: test: simula-ataque-de-60-pb`
  - `:memo: docs: dicionario-das-bases`
  - `:card_file_box: dados: janelas-placebo`
  - `:recycle: refactor: separa-bootstrap-por-ataque`
  
  Escolha o emoji pela lista do gitmoji.dev conforme o tipo da mudança; um assunto por commit.

## 10. Como o Claude deve trabalhar aqui

- **Não invente dados, colunas ou resultados.** Se uma coluna não existe, leia o arquivo e pergunte.
- **Não mude a especificação em silêncio.** Se algo do documento parecer errado ou impossível, explique e pergunte.
- Antes de reportar números, rode as checagens de integridade da base (linhas por ataque e moeda, velas ausentes,
  duplicatas, preços ≤ 0, máxima < mínima, minuto 0 batendo com o horário do anúncio).
- **Peça confirmação antes de chamadas pagas à API em lote**, dizendo quantas notícias e rodadas serão enviadas.
- Não commite `.env`, `results/` ou arquivos acima de 50 MB.
- Ao terminar uma tarefa, diga o que foi feito, como testar e o que ainda falta, em poucas linhas.

## 11. Decisões em aberto

- [ ] Quais dos 9 "achado, conferir" e dos 8 horários suspeitos entram na amostra.
- [ ] Critério final de primeiro anúncio.
- [ ] Quem baixa as janelas placebo (as janelas de ataque da v19 já estão completas).
- [x] ~~Modelo simples para H2 além de "vender sempre"~~ → H2 = b na regressão da CAR + backtest contra o modelo base (06/11).
- [ ] Data de entrega de `dados/llm/scores.csv` (até lá, scores simulados).
- [x] ~~Modelo principal~~ → event study com CAR; VAR-X com faixas vira robustez (07/11).
- [x] ~~Janela de negociação~~ → ataques de 2026; treino = 2020–2025 (07/11).
- [ ] Treino fixo (só 2020–2025) ou expansivo ao longo de 2026 no backtest.
- [ ] Saída base: vale ou meia-vida.
- [ ] Taxa spot da Binance em cada período.
- [ ] Divisão do código: quem faz modelo e backtest (o LLM/score já tem dono).
