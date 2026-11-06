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
| H2 — Ganho do LLM | ler a notícia com o LLM seleciona melhor os ataques do que "vender sempre"? | calibração e backtest |
| H3 — Look-ahead | o LLM acerta porque reconhece o ataque, e não porque entende a notícia? | nome do protocolo mascarado; ataques após o corte de conhecimento do modelo |

## 2. Estrutura do repositório

```
dados/
  eventos/    coleta_tweets_ataques_v19.xlsx — primeiro anúncio de cada ataque
  precos/     minuto_janela_ataques.csv — velas de 1 min, ±3h em torno de cada anúncio
  placebo/    janelas sem ataque (mesmo horário, uma semana antes)
  cci10/      série diária e pesos do CCI10
  llm/        respostas do LLM salvas (uma por notícia e rodada), para nunca chamar a API duas vezes
config/       parâmetros em YAML (ordem de Cholesky, latência, custos...). Nada disso fica fixo no código.
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
- **Saem só nos testes de robustez:** E583, rug pulls e anúncios não confirmados.
- **Pontos em aberto na base v19** (não decida sozinho; sinalize ao grupo):
  - 8 ataques com horário a mais de 2 dias antes ou 3 dias depois da data do DeFiLlama:
    E103, E089, E141, E347, E257, E010, E259, E519.
  - Possíveis não-ataques entre os "conferir": E578 (CoinDCX nega o hack), E413 (Bitget, negociação anormal),
    E382 (post da Arkham).
  - 4 horários com segundos zerados (ex.: E257 às 20:20:00), que parecem digitados à mão.
  - 20 primeiros anúncios já mencionam reembolso ou desfecho (`menciona_desfecho = S`).
- `minuto_janela_ataques.csv` só tem preços para os ataques antigos. Os ataques novos da v19 precisam ter as janelas
  baixadas (Binance spot, velas de 1 min, `data.binance.vision`) **no mesmo formato**.

## 4. A. Modelo de impulso-resposta

**VAR-X empilhado**, um único modelo com todas as janelas:

```
y_{e,t} = α_e + Σ_{j=1..p} A_j y_{e,t-j} + b D_{e,t} + u_{e,t},     D_{e,t} = 1{t = +1}
```

- `y_{e,t}`: vetor dos 10 retornos do ataque e no minuto t. `α_e`: efeito fixo do ataque. `b`: impacto do ataque (10×1).
- **Empilhar** = colocar as janelas uma embaixo da outra e estimar um só conjunto de coeficientes.
- **Defasagens não cruzam janelas:** as p primeiras velas de cada janela entram só como defasagem, nunca como
  variável dependente.
- **Efeito fixo:** tire a média de Y e de X dentro de cada janela (estimador within) e estime por MQO, equação a equação.
- **p por BIC**, comparando todos os candidatos **na mesma amostra** (descarte `max_p` velas de cada janela para todos).
- **Cholesky por tamanho:** BTC, ETH e depois as demais por valor de mercado, a mesma ordem em todo o projeto
  (vem de `config/`). `Σ_u = P P'`.

**Respostas** (h = 0 na vela +1):

```
Ψ_0 = I,  Ψ_h = Σ_{j=1..min(h,p)} Ψ_{h-j} A_j
φ_h   = Ψ_h b            efeito do ataque em cada moeda (pb por minuto)
CR(h) = Σ_{s≤h} φ_s      efeito acumulado = efeito no preço  ← é este que importa para operar
Θ_h   = Ψ_h P            propagação de um choque ortogonal entre as moedas
c     = P⁻¹ b            impacto do ataque em choques ortogonais; φ_h = Θ_h c
```

- φ, CR, vale e meia-vida **não dependem** da ordem de Cholesky. Θ e c dependem: a leitura "por canal" precisa dos
  testes de ordem invertida e IRF generalizada.
- **Intervalos:** bootstrap **por ataque** (reamostra janelas inteiras com reposição, 1.000 réplicas), nunca por minuto.
  Fixe o p do BIC da amostra original. IC percentil de 90%.
- **Placebo:** mesmo modelo nas janelas placebo; o efeito deve ser nulo.
- **Entrega do modelo:** tabela por moeda com CR nas velas **+1, +2, +5, +10, +15, +30, +60 e +180**, minuto do vale e
  meia-vida, cada um com IC, ao lado do placebo; e um gráfico da curva CR completa por moeda.

## 5. Prazo de saída: vale e meia-vida

Com a venda na vela h_e (base: vela +2, ou seja, índice h_e = 1):

```
h_vale = argmin_{h ≥ h_e} CR(h)
h_1/2  = min{ h ≥ h_e : CR(h) − CR(h_vale) ≤ 0,5 [CR(h_e) − CR(h_vale)] }
```

- Nos dois casos, a condição precisa valer por **3 minutos seguidos**.
- Cuidado com a conversão entre **vela** (+1, +2, ...) e **índice h** (0, 1, ...): `h = vela − 1`.
- O vale é instável quando o fundo é plano e cai no fim da janela (+180) se não há reversão. A meia-vida captura
  metade da queda, mas é mais estável. O backtest testa as duas.

## 6. B. LLM

- **Entrada:** só `texto_original` e a data. **Nunca** `valor_roubado_usd` (vem do DeFiLlama, conhecido depois).
- **Saída em JSON**, validada:

| Campo | Tipo |
| --- | --- |
| `confirma_ataque` | booleano |
| `gravidade` | inteiro de 1 a 5, segundo uma régua escrita com exemplos |
| `valor_citado` | número ou null (só se a própria notícia cita) |
| `tipo_ataque` | categoria |
| `tipo_alvo` | categoria (corretora, protocolo DeFi, bridge...) |
| `moedas_expostas` | lista de moedas do CCI10 |
| `confianca` | número |

- O **prompt e a régua ficam em arquivo versionado** e são **congelados antes do teste**. Mudou o prompt, é outro
  experimento.
- Rodar cada notícia de **3 a 5 vezes** e usar a **mediana** da gravidade.
- Validar a régua com 30 a 40 notícias rotuladas à mão pelo grupo.
- **H3:** uma versão com o nome do protocolo mascarado; separar ataques posteriores ao corte de conhecimento do modelo.
- **Técnica:** SDK `anthropic`, saída estruturada e Batch API. Conferir `stop_reason` (textos sobre ataques podem ser
  recusados). A chave fica em `.env` (`ANTHROPIC_API_KEY`) e **nunca** vai para o Git.
- Salvar toda resposta bruta em `dados/llm/` e ler do cache antes de chamar a API.

**Calibração** (só com ataques de treino):

```
queda_{e,i} = a + b·gravidade_e + c·exposta_{e,i} + ε,     exposta_{e,i} = 1{ i ∈ moedas_expostas_e }
```

`queda` é o retorno acumulado realizado da moeda i entre a venda e a saída, em pb (negativo quando cai).

## 7. C. Backtest

- **Janela expansiva:** para cada ataque a partir do ~30º, reestimar com **todos os ataques anteriores** os coeficientes
  do VAR-X (logo, o vale e a meia-vida) e a regressão de calibração; aplicar ao ataque atual.
- **Fica fixo** ao longo do backtest: prompt e régua do LLM, ordem de Cholesky, vela do ataque, regra de saída,
  custos e o p escolhido na primeira estimação.
- **Regra:** vender a moeda i se `−queda_prevista_{e,i} > custo de ida e volta`. `confirma_ataque = false` → não opera.
- **Venda** na vela +2 (testar +1, +5, +10). **Recompra** no vale ou na meia-vida da moeda.
- **Prêmio** de cada ataque e retorno da estratégia:

```
π_e = Σ_{i vendidas} w_i^{CCI10} (1 − P_{i,recompra} / P_{i,venda}) − custos_e
R_e(estratégia) = R_e(CCI10) + π_e
```

- **Custos:** taxa de ~0,10% por lado (confirmar na Binance), spread e impacto; ordem limitada a 5–10% do volume do
  minuto; carteiras de US$ 100 mil, 1 mi e 10 mi.
- **Comparações:** CCI10 parado; vender sempre (sem LLM); vender em datas placebo; previsão perfeita.
- **Métricas:** prêmio por operação e acumulado, taxa de acerto, Information Ratio, drawdown máximo, teste t e
  Sharpe deflacionado (Bailey e López de Prado, 2014).
- Versão rápida, se faltar tempo: estimar uma vez com os ataques até 2024 e aplicar a 2025–26.

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
- **Testes com dados simulados** em que a resposta é conhecida: por exemplo, simular um VAR(1) com um ataque de −60 pb
  no BTC e conferir que o modelo recupera A e b; conferir que as defasagens não cruzam janelas; conferir que o
  backtest nunca vê ataques futuros.
- Fluxo Git: uma branch por tarefa (`modelo/`, `llm/`, `backtest/`, `dados/`), Pull Request revisado por outra
  pessoa do grupo, testes verdes antes do merge.
- **Mensagens de commit (obrigatório)** seguem o [gitmoji](https://gitmoji.dev), no formato
  `<emoji> comando: nome-do-commit`, com o emoji em código (`:sparkles:`), em minúsculas e em português. Exemplos:
  - `:sparkles: feat: estima-var-x-empilhado`
  - `:bug: fix: defasagens-nao-cruzam-janelas`
  - `:white_check_mark: test: simula-var1-com-ataque`
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
- [ ] Quem baixa as janelas de preço dos ataques novos e as janelas placebo.
- [ ] Modelo simples para H2 além de "vender sempre".
- [ ] Saída base: vale ou meia-vida.
- [ ] Taxa spot da Binance em cada período.
- [ ] Divisão do código: quem faz modelo, LLM e backtest.
