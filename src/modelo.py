"""Event study intradiário (CLAUDE.md, seções 4 e 5): retorno anormal acumulado (CAR) por ataque e moeda,
curva média CR(h), modelo com score CR(h; g), vale, meia-vida e bootstrap por ataque.

Convenções
  vela t    = minuto relativo ao anúncio, de −180 a +180 (vela 0 = minuto do anúncio).
  r[..., j] = log-retorno da vela t = j − 179, em pb: 1e4 (ln P_t − ln P_{t−1}). Sai da vela −179 até +180,
              porque a vela −180 não tem anterior dentro da janela.
  h         = índice a partir da vela +1 (h = vela − 1), de 0 a 179.
Todas as curvas e CARs estão em pb.
"""
import numpy as np
import pandas as pd

T0 = -180  # primeira vela da janela


def carrega_janelas(caminho, coluna="close"):
    """Lê minuto_janela_ataques.csv(.gz) ou minuto_janela_placebo.csv.gz.

    coluna = "close" (padrão) ou outra coluna da base, ex. "quote_volume" (volume em USDT).
    Sai: close [E, I, 361] (a coluna pedida), eventos (DataFrame com event_id e hora_referencia_utc, na ordem
    de close) e moedas (lista, na ordem de close). Falha se faltar vela ou se houver preço ≤ 0 ou duplicata.
    """
    df = pd.read_csv(caminho, usecols=["event_id", "hora_referencia_utc", "moeda", "minuto_relativo", coluna])
    if df.duplicated(["event_id", "moeda", "minuto_relativo"]).any():
        raise ValueError(f"{caminho}: linhas repetidas")
    if coluna == "close" and (df.close <= 0).any():
        raise ValueError(f"{caminho}: preço ≤ 0")
    eventos = df.drop_duplicates("event_id")[["event_id", "hora_referencia_utc"]].sort_values("event_id")
    eventos["hora_referencia_utc"] = pd.to_datetime(eventos.hora_referencia_utc)
    moedas = sorted(df.moeda.unique())
    grade = pd.MultiIndex.from_product([eventos.event_id, moedas, range(-180, 181)])
    s = df.set_index(["event_id", "moeda", "minuto_relativo"])[coluna].reindex(grade)
    if s.isna().any():
        raise ValueError(f"{caminho}: {s.isna().sum()} velas faltando")
    return s.to_numpy().reshape(len(eventos), len(moedas), 361), eventos.reset_index(drop=True), moedas


def calcula_car(close, normal=(-180, -61), pre=(-10, 0)):
    """CAR por ataque e moeda com retorno normal de média constante.

    Entra: close [E, I, 361]; normal = (vela inicial, vela final) do retorno normal; pre = faixa da CAR pré.
           A vela −180 não tem retorno, então uma janela normal que começa em −180 usa de −179 em diante.
    Sai:   dict com mu [E, I] (pb/min), ar [E, I, 360] (pb, velas −179..+180), car [E, I, 180]
           (pb, CAR desde a vela +1: car[..., h] = soma do AR das velas 1..h+1) e car_pre [E, I] (pb).
    """
    r = 1e4 * np.diff(np.log(close), axis=-1)
    j = lambda vela: vela - T0 - 1  # posição da vela em r
    mu = r[..., j(max(normal[0], T0 + 1)): j(normal[1]) + 1].mean(axis=-1)
    ar = r - mu[..., None]
    return {"mu": mu, "ar": ar,
            "car": np.cumsum(ar[..., j(1):], axis=-1),
            "car_pre": ar[..., j(pre[0]): j(pre[1]) + 1].sum(axis=-1)}


def retorno_bruto(close, janelas=((-60, -11), (-10, 0))):
    """Retorno bruto (sem μ) desde a vela 0 e movimentos pré-anúncio, para o modelo condicional.

    Entra: close [E, I, 361]; janelas = faixas (vela inicial, vela final) do movimento pré-anúncio.
    Sai:   y [E, I, 180] (pb, y[..., h] = 1e4 ln(P_{vela h+1} / P_{vela 0})) e
           x [E, I, K] (pb, x[..., k] = 1e4 ln(P_fim / P_{inicio−1}), soma dos retornos das velas da faixa k).
    Sem μ de propósito: com AR, preditor e drift descontam o mesmo μ e o erro dele cria correlação espúria
    (aparece igual no placebo).
    """
    lp = 1e4 * np.log(close)
    p = lambda vela: vela - T0
    y = lp[..., p(1):] - lp[..., p(0), None]
    x = np.stack([lp[..., p(fim)] - lp[..., p(ini - 1)] for ini, fim in janelas], axis=-1)
    return y, x


def regressao_condicional(y, g, x):
    """MQO empilhado y_{e,i}(h) = a_i(h) + b(h)(g_e − ḡ) + Σ_k c_k(h)(x_{e,i,k} − x̄_{i,k}) + ε, por h.

    Entra: y [E, I, H] (pb); g [E] (score 1–5); x [E, I, K] (pb, preditores por ataque e moeda).
    Sai:   a [I, H] (pb, média de y por moeda = efeito do ataque médio), coef [1+K, H] (b em pb por ponto de score,
           c_k em pb por pb) e zbar [I, 1+K] (médias usadas para centrar: ḡ e x̄_i).
    Inclinações comuns às moedas; centrar por moeda equivale à dummy de moeda. Curva de um ataque novo com
    z = [g, x_1..x_K] por moeda ([I, 1+K]): a + (z − zbar) @ coef.
    """
    z = np.concatenate([np.broadcast_to(g[:, None, None], (*x.shape[:2], 1)), x], axis=-1)
    zbar, a = z.mean(0), y.mean(0)
    coef = np.linalg.lstsq((z - zbar).reshape(-1, z.shape[-1]), (y - a).reshape(-1, y.shape[-1]), rcond=None)[0]
    return a, coef, zbar


def pesos_bootstrap(n, replicas, seed):
    """Contagens [replicas, n]: quantas vezes cada ataque entra em cada réplica (reamostragem com reposição)."""
    idx = np.random.default_rng(seed).integers(0, n, (replicas, n))
    c = np.zeros((replicas, n))
    np.add.at(c, (np.arange(replicas)[:, None], idx), 1)
    return c


def media_ponderada(y, pesos):
    """Média de y [E, ...] sobre os ataques, uma por linha de pesos [R, E]. Sai [R, ...]."""
    return (pesos @ y.reshape(len(y), -1)).reshape(len(pesos), *y.shape[1:]) / pesos.sum(1).reshape(-1, *[1] * (y.ndim - 1))


def regressao_score(car, g, pesos=None):
    """MQO por moeda e por h de CAR_{e,i}(h) = a_i(h) + b_i(h) (g_e − ḡ) + ε.

    Entra: car [E, I, H] (pb); g [E] (score 1–5); pesos [R, E] (contagens do bootstrap) ou None (amostra original).
    Sai:   a, b com forma [I, H] (sem pesos) ou [R, I, H]; e gbar (escalar ou [R]). ḡ é recalculado em cada réplica.
    """
    p = np.ones((1, len(g))) if pesos is None else pesos
    gbar = p @ g / p.sum(1)
    x = g[None, :] - gbar[:, None]                                           # [R, E]
    y = car.reshape(len(g), -1)
    a = (p @ y) / p.sum(1)[:, None]
    b = ((p * x) @ y) / (p * x ** 2).sum(1)[:, None]
    forma = (len(p), *car.shape[1:])
    a, b = a.reshape(forma), b.reshape(forma)
    return (a[0], b[0], gbar[0]) if pesos is None else (a, b, gbar)


def inclinacao_comum(b):
    """b comum às moedas: média de b_i(h) entre as moedas, repetida em cada moeda.

    Entra: b [I, H] ou [R, I, H] (pb por ponto de score), saído de regressao_score. Sai: mesma forma.
    Com painel balanceado (todas as moedas com os mesmos ataques e o mesmo g), isso é exatamente o MQO empilhado
    CAR_{e,i}(h) = a_i(h) + b(h) (g_e − ḡ) + ε, com dummy de moeda: x é o mesmo em toda moeda, então o
    numerador da inclinação é a soma dos numeradores por moeda e o denominador é I vezes o de uma moeda.
    """
    return np.broadcast_to(b.mean(-2, keepdims=True), b.shape)


def b_comum_huber(car_h, g, pesos=None, t=1.345):
    """b comum às moedas por regressão robusta de Huber (robustez do MQO), em uma vela.

    Entra: car_h [E, I] (pb, CAR de uma vela); g [E] (score 1–5); pesos [R, E] (contagens do bootstrap) ou None;
           t = constante de Huber em desvios-padrão robustos (1,345 → ~95% de eficiência sob erro normal).
    Sai:   b (pb por ponto de score), escalar ou [R].
    Modelo empilhado (E·I linhas): CAR_{e,i} = a_i + b (g_e − ḡ) + ε, com dummy de moeda, estimado por RLM com
    HuberT e escala MAD. No bootstrap, cada ataque entra tantas vezes quanto a contagem dele.
    """
    import statsmodels.api as sm
    E, I = car_h.shape

    def ajusta(idx):
        x = np.repeat(g[idx] - g[idx].mean(), I)                         # linhas: ataque por fora, moeda por dentro
        X = np.column_stack([np.tile(np.eye(I), (len(idx), 1)), x])
        return sm.RLM(car_h[idx].ravel(), X, M=sm.robust.norms.HuberT(t)).fit().params[-1]

    if pesos is None:
        return ajusta(np.arange(E))
    return np.array([ajusta(np.repeat(np.arange(E), c.astype(int))) for c in pesos])


def vale_meia_vida(cr, h_venda=1, seguidos=3):
    """Vale e meia-vida de uma ou várias curvas CR (seção 5), em índice h.

    Entra: cr [..., H] (pb); h_venda = índice da vela de venda (vela +2 → h = 1); seguidos = minutos seguidos.
    Sai:   (h_vale, h_meia), inteiros com a forma de cr[..., 0].
      vale  = início do bloco de `seguidos` minutos (h ≥ h_venda) cujo máximo é o mais baixo; fundo = esse máximo.
      meia  = primeiro h ≥ h_venda em que as `seguidos` velas ficam ≤ fundo + 0,5 [CR(h_venda) − fundo].
      Sem queda depois da venda (fundo ≥ CR(h_venda)): vale = meia = h_venda.
    """
    jan = np.lib.stride_tricks.sliding_window_view(cr[..., h_venda:], seguidos, axis=-1).max(-1)  # [..., B]
    k = jan.argmin(-1)
    fundo = np.take_along_axis(jan, k[..., None], -1)[..., 0]
    inicio = cr[..., h_venda]
    sem_queda = fundo >= inicio
    alvo = fundo + 0.5 * (inicio - fundo)
    meia = (jan <= alvo[..., None]).argmax(-1)  # existe sempre: o bloco do vale satisfaz
    return (np.where(sem_queda, h_venda, k + h_venda), np.where(sem_queda, h_venda, meia + h_venda))


def intervalo(reps, nivel=0.90):
    """IC percentil das réplicas [R, ...]. Sai (inferior, superior)."""
    q = (1 - nivel) / 2
    return np.quantile(reps, q, axis=0), np.quantile(reps, 1 - q, axis=0)
