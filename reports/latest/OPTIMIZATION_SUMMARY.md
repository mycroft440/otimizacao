# Otimização exaustiva — B3 Estratégia Live

**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`
**Combinações testadas:** 1.112.032
**Período solicitado:** 2018-01-02 até 2026-09-02
**Curva observada:** 2018-01-02 até 2026-09-02
**Universo:** 40 ações
**Carteira:** Top1 semanal; mesmo Top1 = manutenção integral, sem nova ordem.
**Reconciliação motor rápido × replay detalhado:** PASS.

## Melhor combinação por patrimônio final

- GAP_PERIOD: **41**
- SIGNAL_PERIOD: **15**
- MOMENTUM_PERIOD: **49**
- VOL_PERIOD: **21** (fixo)
- Capital inicial: **R$ 1000.00**
- Patrimônio mark-to-market final: **R$ 89280.33**
- Retorno total: **8828.03%**
- CAGR: **67.93%**
- Média dos anos completos: **60.04%**
- Max drawdown close-to-close: **-44.33%**
- Sharpe rf=0 close-to-close: **1.4367**
- Trades: **491**
- Execuções puladas: **0**
- Taxas: **R$ 2511.33**
- Slippage: **R$ 8067.39**

### Retorno por ano

- 2018: **76.94%**
- 2019: **120.11%**
- 2020: **69.66%**
- 2021: **89.01%**
- 2022: **9.49%**
- 2023: **27.06%**
- 2024: **25.14%**
- 2025: **62.94%**
- 2026 (parcial): **152.00%**

## Pine original 40/20/63

- Patrimônio final: **R$ 10561.35**
- Retorno total: **956.14%**
- CAGR: **31.26%**
- Vantagem patrimonial da vencedora: **745.35%**

## Metodologia

Busca exaustiva: GAP 5–80, Signal 2–60 e Momentum 5–252, passo 1; Vol21 fixo.

O sinal usa o fechamento da última sessão da semana anterior e a execução usa a abertura da primeira sessão B3 seguinte. Quando o Top1 não muda, nenhuma ordem é criada. Quando muda, a operação é atômica: se venda ou nova compra não puder ser executada, a carteira anterior permanece intacta.

Métricas de risco usam a curva diária marcada a fechamento; Sharpe usa taxa livre de risco zero e anualização por sqrt(252).

**Atenção:** a vencedora continua sendo in-sample; a validação OOS/walk-forward é separada.

## Auditoria endurecida de metricas e valor terminal

- Mark-to-market final: **R$ 89280.33**
- Liquidation equity estimada: **R$ 89161.71**
- Custo estimado para liquidar a posicao terminal: **R$ 118.62**
- Max drawdown reportado: **-44.33%**, medido close-to-close diario.
- Sharpe: **1.4367**, risk-free=0 e anualizacao sqrt(252).
- Anos iniciais e terminais parciais nao entram na media de anos completos.
