# Explicação do modelo

Situação em 07/11/2026 (com a divisão treino/negociação). Este arquivo explica o modelo e as decisões tomadas pelo
grupo; não descreve código.

---

# Como o modelo funciona

## 1.1 A pergunta

Quando sai o **primeiro tweet** anunciando que um protocolo cripto foi hackeado, o que acontece com o preço das
10 moedas do CCI10 (BTC, ETH, XRP, BNB, ADA, LINK, XLM, TRX, LTC, BCH) nos minutos seguintes?

- Se o preço cai **tudo de uma vez**, no minuto do tweet, ninguém consegue aproveitar: quando você lê, já caiu.
  Isso é o "mercado eficiente".
- Se o preço cai **aos poucos** ao longo de vários minutos (o *drift*), existe uma janela em que daria para
  vender logo depois do tweet e recomprar mais barato.

O modelo mede **quanto** cai, **em que ritmo** e **até quando**.

## 1.2 Os dados e o vocabulário

| Termo | O que é |
| --- | --- |
| **Ataque** | um hack listado na planilha de eventos. Há 196 com o horário exato do primeiro tweet (eram 197; o E519 saiu por ser o mesmo incidente do E524): **151 de 2020 a 2025 (treino)** e **45 de 2026 (negociação)**. Ver 1.4. |
| **Minuto 0** | o minuto em que o tweet saiu, arredondado para baixo. Tweet às 14:03:27 → minuto 0 = 14:03. |
| **Vela** | o preço de um minuto (abertura, máxima, mínima, fechamento), da Binance. A **vela +1** é o primeiro minuto inteiro depois do tweet (14:04 no exemplo). A **vela −1** é o minuto antes do tweet. |
| **Janela** | os 361 minutos em torno de um ataque: de 3 horas antes (−180) a 3 horas depois (+180). Cada ataque tem uma janela para cada moeda. |
| **Retorno** | quanto o preço variou de um minuto para o seguinte. |
| **pb (ponto-base)** | 1 pb = 0,01%. Se o BTC vai de 100.000 para 99.900, o retorno foi −10 pb. Tudo no modelo é em pb. |

O retorno é calculado **dentro** de cada janela. Nunca se calcula o retorno entre o último minuto de um ataque e o
primeiro minuto de outro ataque meses depois.

## 1.3 O placebo

Imagine que medimos uma queda média de 15 pb depois dos ataques. Isso pode ter acontecido por três motivos:

1. o ataque derrubou o preço;
2. o mercado cripto cai uns 15 pb em qualquer período de uma hora;
3. aqueles horários do dia costumam ser ruins.

O **placebo** separa o motivo 1 dos outros dois. Para cada ataque, pegamos o **mesmo dia da semana e o mesmo
horário, 7 dias antes**, quando não houve ataque, e fazemos **exatamente a mesma conta**. Se essa janela cai em
cima de outro ataque (6 casos), usamos 2 semanas antes (ou 3, ou 4).

- Se o placebo também cai 15 pb, a queda não tem nada a ver com o ataque.
- Se o placebo dá perto de zero e o ataque dá −15 pb, a queda é do ataque.

É o mesmo papel do comprimido de farinha num teste de remédio. O placebo usa as mesmas 196 datas deslocadas em uma
semana, as mesmas 10 moedas e o mesmo formato de dados.

## 1.4 Treino e negociação: por que o modelo não vê 2026

O grupo decidiu que a **negociação** (o teste da estratégia) acontece nos **45 ataques de 2026**. Daí vem uma regra
que não pode ser quebrada: **o modelo só pode aprender com os ataques de antes de 2026.**

O motivo: se o modelo aprendesse com os ataques de 2026 e depois "operasse" esses mesmos ataques, ele já saberia o
que ia acontecer. Seria como estudar para a prova com o gabarito. O resultado ficaria ótimo e não valeria nada.

| Período | Ataques | Para que serve |
| --- | --- | --- |
| **Treino** (2020 a 2025) | 151 | estimar o modelo: a curva CR, o efeito da nota, a nota média, o bootstrap |
| **Negociação** (2026) | 45 | só para os colegas do backtest aplicarem o modelo já pronto; o modelo nunca vê esses preços |

A fronteira está limpa: o último ataque do treino é de 27/12/2025 e a janela dele termina muito antes do primeiro
ataque de 2026 (08/01). Nenhuma janela de treino pode invadir 2026.

**Um resultado importante que já aparece nos dados brutos** (média simples das 10 moedas, de +1 a +60):

| | Ataque | Placebo |
| --- | --- | --- |
| Treino (2020–2025, 151 ataques) | −21,8 pb (t = −2,5) | −2,1 pb |
| Negociação (2026, 45 ataques) | +4,9 pb (t = +0,6) | +5,3 pb |

A queda depois dos ataques está **toda no treino**. Em 2026 ela não aparece. Com 45 ataques isso ainda pode ser
acaso, mas também é consistente com o mercado ter ficado mais eficiente. Para a estratégia é uma má notícia: o
modelo, treinado até 2025, vai prever uma queda que em 2026, na média, não aconteceu. Para o trabalho é um bom
resultado, porque é exatamente o tipo de evidência que sustenta o argumento de eficiência de mercado.

Qualquer parâmetro do modelo (janela do retorno normal, vela de venda, regra de saída) tem que ser congelado antes
de rodar o backtest e não pode mudar depois de ver o resultado.

## 1.5 A ideia central: quanto caiu além do normal

O modelo é um **event study** (estudo de evento), o método clássico de finanças para medir a reação do preço a uma
notícia (MacKinlay, 1997). A ideia: para cada ataque e cada moeda, comparar o que o preço fez depois do tweet com o
que ele faria num minuto normal.

```
retorno anormal = retorno do minuto − retorno normal
```

### Passo 1 — Retorno normal

Cada janela tem o seu "clima": num dia o mercado estava subindo, em outro caindo. O **retorno normal** é a média dos
retornos da própria janela entre o minuto −180 e o −61, longe do anúncio. A última hora antes do tweet fica de fora
porque o ataque já está acontecendo na blockchain e a notícia pode vazar antes: nos dados de treino a queda começa
antes do minuto −11. Se essa queda entrasse no "normal", o modelo acharia que cair é normal e atribuiria ao ataque
só parte da queda depois do tweet.

### Passo 2 — Retorno anormal (AR)

O **AR** de um minuto é o retorno daquele minuto menos o retorno normal. Exemplo: o BTC costuma render +0,1 pb por
minuto naquela janela. No minuto +3 ele caiu −4 pb. O AR desse minuto é −4,1 pb, a parte "culpa do ataque".

### Passo 3 — Retorno anormal acumulado (CAR)

A **CAR** é a soma dos AR desde a vela +1 até o minuto que interessa. Se os AR forem −5, −3, −2 e +1 pb nos minutos
+1 a +4, a CAR em +4 é −9 pb: até ali a moeda caiu 9 pb a mais do que cairia sem o ataque.

Há também a **CAR pré**: a soma dos AR de −10 até a vela 0. Ela serve para entender o fenômeno (a notícia vaza
antes do tweet), mas **não dá para operar** com isso, porque antes do tweet ninguém sabe do ataque.

### Por que não usamos faixas

A versão anterior do modelo (um VAR-X) agrupava os minutos em faixas (−10 a −1, 0, +1, +2 a +5, +6 a +15, +16 a +60).
As faixas foram escolhidas olhando a média dos ataques, o que é fácil de criticar, e a curva ficava reta depois de
+60 por construção. No event study **cada minuto entra separado**: não há faixas para escolher e a curva tem a forma
que os dados mostrarem, até +180. O VAR-X continua como teste de robustez.

## 1.6 Da CAR de cada ataque à curva CR

A curva **CR** é a **média das CAR** dos 151 ataques de treino, moeda a moeda e minuto a minuto. É ela que diz
quanto o preço cai, em média, até cada momento depois do tweet.

`CR na vela +60 = −19 pb` no ETH quer dizer: do tweet até 1 hora depois, o ETH caiu em média 19 pb a mais do que
cairia sem o ataque.

**Propagação entre as moedas (H1):** comparamos a CR de cada moeda (quanto cai; as menores caem mais?) e o minuto em
que cada uma chega a metade da queda (quando cai; o BTC reage primeiro?). Isso responde se a queda se espalha e em
que ordem, sem precisar de um modelo de "quem puxa quem".

**Cuidado com ataques próximos:** três pares de ataques aconteceram a menos de 3 horas um do outro (E155–E156,
E290–E288, E391–E393). A janela de um contém o outro, então a robustez roda sem esses pares.

## 1.7 Intervalos de confiança (bootstrap)

Com 151 ataques de treino, todo número tem incerteza. Para medi-la usamos o **bootstrap por ataque**:

1. Sorteamos 151 ataques da lista **com reposição**: alguns saem duas vezes, outros nenhuma.
2. Refazemos a conta inteira com esse sorteio.
3. Repetimos isso **1.000 vezes**.

Os 1.000 resultados mostram o quanto o número "balança". O **intervalo de 90%** é a faixa onde caem os 90% do meio.

- O intervalo **não inclui zero** → o efeito é confiável.
- O intervalo **inclui zero** → pode ser acaso.

O sorteio é de **ataques inteiros**, nunca de minutos soltos, porque os minutos de um mesmo ataque são parecidos
entre si. A seed é fixa, então rodar de novo dá exatamente o mesmo resultado.

## 1.8 Vale e meia-vida: quando a queda acaba

Pela especificação, a venda acontece na **vela +2**: um minuto para ler a notícia e agir. A pergunta seguinte é
quando recomprar. Calculamos dois candidatos sobre a curva CR de cada moeda:

- **Vale:** o minuto em que a curva chega ao ponto mais baixo depois da venda, ou seja, onde a queda termina.
- **Meia-vida:** o primeiro minuto em que já aconteceu **metade** da queda entre a venda e o vale. É uma saída mais
  cedo e mais conservadora.

Nos dois casos a condição tem que valer por **3 minutos seguidos**, para que um minuto isolado de ruído não seja
confundido com o fundo. No vale, isso foi lido como "o início do bloco de 3 minutos com o fundo mais baixo"; o
grupo ainda precisa confirmar essa leitura.

Como a curva do event study é minuto a minuto, ela é mais ruidosa que a do VAR-X. O vale pode oscilar entre minutos
com quedas quase iguais; a **meia-vida** tende a ser mais estável.

## 1.9 Onde entra a nota do LLM (modelo com score)

Existem **dois modelos**:

| | Modelo base | Modelo com score |
| --- | --- | --- |
| Usa a nota do LLM? | não | sim |
| Pergunta que responde | H1: o ataque derruba o preço? | H2: ataques mais graves derrubam mais? |
| Conta | média das CAR | regressão da CAR na nota |

O LLM lê o texto do primeiro tweet e dá uma **nota de gravidade de 1 a 5**. No modelo com score, para cada moeda e
cada minuto, fazemos uma regressão da CAR dos ataques na nota:

```
CAR do ataque = a + b × (nota do ataque − nota média) + erro
```

- **a** é a queda de um ataque de gravidade média. Como a nota é centrada, `a` é igual à CR do modelo base.
- **b** é quanto a queda muda a cada ponto de nota acima ou abaixo da média. Se b é negativo e o intervalo não
  inclui zero, ataques mais graves derrubam mais (H2).
- A **nota média** é a média dos ataques de treino (perto de 3).

Exemplo com números inventados (a = −10 pb e b = −4 pb na vela +15, nota média 3): nota 1 → −2 pb,
nota 3 → −10 pb, nota 5 → −18 pb.

Regras:

- A nota **só existe a partir do tweet**. Ela explica a CAR desde a vela +1, nunca a CAR pré.
- No bootstrap, cada ataque sorteado leva a sua nota junto, e a nota média é recalculada a cada sorteio.
- A relação é **linear**: cada ponto de nota vale o mesmo. Uma regressão por nota evitaria isso, mas sobrariam ~30
  ataques de treino por nota, o que é pouco.

**As notas reais chegaram em 07/11** (`dados/llm/scores.csv`, 1 rodada por notícia). Quase todas ficam entre 2 e
3; só um ataque tem nota 5, então a curva de g = 5 é extrapolada pela reta e tem intervalo muito largo. Se o arquivo
sumir, o modelo volta a sortear notas e marca tudo com `_SIMULADO` no nome; esses números não vão para os slides.

## 1.10 Testes de robustez

- **Modelo de mercado:** em vez da média da janela, o retorno normal vem de uma regressão da moeda no mercado,
  estimada em −180 a −61. Cuidado: se o mercado inteiro cai com o ataque, descontar o mercado tira parte do efeito.
- **VAR-X com faixas** (o modelo anterior): estima os padrões normais de um minuto para o outro e o efeito do
  ataque por faixa. Se a curva dele ficar parecida com a do event study, o resultado fica mais forte.
- **Sem os pares de ataques próximos** e **sem os casos duvidosos** da base (seção 3 do CLAUDE.md).

## 1.11 Resultados preliminares (VAR-X com faixas, código anterior)

**Estes números saíram do VAR-X com faixas, cujo código foi apagado. Precisam ser refeitos com o event study e não
vão para os slides assim.** Efeito acumulado em pb, com intervalo de 90% entre colchetes (1.000 sorteios):

| Moeda | Pré (até a vela 0) | CR +5 | CR +15 | CR +60 | CR +60 no placebo |
| --- | --- | --- | --- | --- | --- |
| BTC | −4,9 [−9,6; −0,4] | −3,6 [−6,2; −1,2] | −4,0 [−8,2; −0,2] | −9,7 [−18,6; −2,3] | −0,7 [−7,5; 5,6] |
| ETH | −6,3 [−12,0; −0,8] | −5,0 [−8,5; −1,9] | −6,3 [−11,9; −1,2] | −19,0 [−34,2; −7,2] | 0,6 [−8,7; 9,1] |
| BNB | −4,8 [−9,7; −0,4] | −4,4 [−9,0; −0,7] | −7,5 [−14,5; −1,6] | −18,8 [−36,9; −5,5] | −4,2 [−12,9; 4,1] |
| LINK | −9,2 [−16,1; −2,6] | −7,5 [−11,8; −2,9] | −15,8 [−25,3; −7,5] | −24,6 [−42,7; −9,3] | 1,6 [−9,9; 13,5] |
| TRX | 2,7 [−5,2; 13,8] | −2,0 [−4,4; 0,4] | −7,9 [−14,5; −1,8] | −20,5 [−31,7; −11,0] | 0,8 [−5,8; 7,0] |
| XRP, ADA, XLM, LTC, BCH | −4 a −10 | −4 a −7 | −7 a −12 | −5 a −17, intervalos encostando ou cruzando o zero | entre −5 e +4 |

Leitura:

1. **O ataque derruba o preço, e a queda é gradual.** Ela continua ao longo da primeira hora. Em BTC, ETH, BNB,
   LINK e TRX, o intervalo de +60 não inclui zero. O placebo fica perto de zero em todas as moedas.
2. **Parte da queda acontece antes do tweet**: −5 a −9 pb até a vela 0 na maioria das moedas. A informação vaza
   antes do anúncio.
3. **A queda média ainda é menor que o custo de operar** (~26 pb ida e volta, estimativa provisória). Isso não
   torna a estratégia inviável, mas sugere que ela só funcionaria nos ataques mais graves. É aí que entra a nota
   do LLM.
4. **A média esconde muita diferença entre ataques**: a mediana da queda bruta é perto de zero. Poucos ataques
   grandes puxam a média.

---

# O que falta fazer

## Estado em 07/11 (noite)

Branch `modelo/janela-normal-e-placebo`, commitada e **ainda sem push nem PR** (já traz a main com os arquivos do LLM).

**Feito**
- Placebo A baixado (`dados/placebo/minuto_janela_placebo.csv.gz`): 196 janelas, 190 a 1 semana e 6 a 2 semanas
  antes (E068, E107, E154, E331, E410, E561, que cruzavam outro ataque). Integridade conferida.
- Retorno normal passou a [−180, −61] (vazamento antes do anúncio contaminava o μ).
- Event study completo em `src/modelo.py` + `scripts/04_modelo.py`: CAR, CR por moeda, bootstrap por ataque, vale,
  meia-vida, modelo com score, placebo, tabelas e gráficos em `results/parametros/`. Testes em `tests/`.
- Arquivos do LLM organizados em `dados/llm/` e `scripts/03_score_llm.py`; `scores.csv` real (1 rodada).

**Primeiros resultados (151 ataques de treino)**
- H1: queda gradual, ~−1 pb na vela +1 e −15 a −35 pb em +60; em BTC, ETH, BNB, LINK e TRX o IC de +60 exclui o
  zero. Placebo perto de zero. Em +180 todos os IC cruzam o zero. CAR pré de −5 a −12 pb.
- H2: o score não explica a CAR. Só 2 de 80 coeficientes `b` (moeda × vela) com IC fora do zero, menos que o acaso.
- Vale muito instável (IC de ~30 a ~170); meia-vida entre as velas 12 e 39.

**Como rodar** (o Python 3.11 da máquina do Lucas é 32 bits e não instala pandas; usar o `.venv` com 3.14):
```
.venv/Scripts/python -m pytest tests
.venv/Scripts/python scripts/04_modelo.py
```

**Próximos passos**
1. Push e PR da branch para revisão.
2. Grupo decidir a saída base
   (vale ou meia-vida).
3. Backtest (seção 7 do CLAUDE.md), ainda não começado.
4. Robustezes do modelo (lista abaixo), cada uma como um YAML em `config/` passado com `--config`.

## Decisões do grupo (padrão provisório)

| Ponto | Padrão hoje |
| --- | --- |
| E519 ou E524 (mesmo incidente) | fica o E524; o E519 saiu da base dos colegas em 07/11 |
| 8 horários suspeitos e 3 possíveis não-ataques | ficam na amostra, só sinalizados |
| Leitura dos "3 minutos seguidos" no vale | início do bloco de 3 minutos com o fundo mais baixo; meia-vida = 1º bloco de 3 minutos todo abaixo da metade da queda; sem queda depois da venda, vale = meia-vida = vela de venda |
| Rodadas do LLM | 1 por notícia (decidido pelo grupo em 07/11; a especificação pedia 3 a 5) |
| Saída base: vale ou meia-vida | os dois são calculados; a meia-vida é mais estável |
| Modelo principal | event study com CAR; VAR-X com faixas só como robustez (decidido em 07/11) |
| Janela do retorno normal | −180 a −61 (decidido em 07/11; congelada antes do backtest) |
| Início da negociação | 01/01/2026 (45 ataques) |

## Dados

- [x] **Notas reais do LLM** (`dados/llm/scores.csv`, 196 ataques, 1 rodada por decisão do grupo).
- [x] **Janelas placebo** (mesmo formato dos ataques, 7 dias antes): `scripts/02_baixa_placebo.py`.

## Modelo

- [ ] **Robustez da média:** conferir a mediana e refazer sem os 5 maiores ataques, para saber se poucos casos
      puxam o resultado.
- [ ] **Testes de robustez da especificação** (seção 3 do CLAUDE.md): sem E583, sem rug pulls, sem anúncios não
      confirmados, sem os horários suspeitos.
- [ ] **Refazer a tabela da seção 1.11** com o event study.
- [ ] **Robustez:** modelo de mercado, VAR-X com faixas e sem os pares de ataques próximos (seção 1.10).
- [ ] **H3** (o LLM acerta porque reconhece o ataque?): depende das notas com o protocolo mascarado e da separação
      pela data de corte do modelo.

## Backtest

- Os ataques a operar são os de 2026; o modelo é estimado só com 2020–2025. Se o grupo reestimar ao longo de 2026
  (janela expansiva), cada ataque de 2026 já operado pode entrar no treino dos seguintes, mas **nunca** no dele
  mesmo. Treino fixo ou expansivo ainda é decisão em aberto.
- Cuidado: três pares de ataques aconteceram a menos de 3 horas um do outro (E155–E156, E290–E288, E391–E393). Se
  um deles entra no treino, a janela dele contém preços de depois do outro ataque, o que é informação do futuro.
  Ataques cuja janela ainda não terminou no minuto 0 do ataque operado devem ficar fora do treino.
