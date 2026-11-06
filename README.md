# Datathon 2026 — Drift pós-ataque

Como o primeiro anúncio de um ataque hacker se propaga, minuto a minuto, entre as 10 moedas do CCI10, e se vale
vender para USDT depois da notícia e recomprar quando o efeito passa.

- **Bases de dados:** `dados/` (dicionário em `dados/README.md`)
- **Especificação e regras para o código:** `CLAUDE.md`
- **Documentos do grupo:** `docs/README.md`

## Começar

```bash
git clone https://github.com/fgvquant/datathon-2026.git
cd datathon-2026
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # e coloque a chave da API do LLM
```

Para programar com o Claude Code, abra o terminal na pasta do repositório e rode `claude`: ele lê o `CLAUDE.md`
automaticamente.

## Cronograma

| Dia | Entrega |
| --- | --- |
| Sexta, 06/11 | Modelo de impulso-resposta e placebo |
| Sábado, 07/11 | LLM e calibração |
| Domingo, 08/11 | Backtest e slides, até 23:59 |
