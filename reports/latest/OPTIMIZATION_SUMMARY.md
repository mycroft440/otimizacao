# Otimização exaustiva — B3 Estratégia Live

**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`
**Combinações testadas:** 1.112.032
**Período:** 2018-01-02 até 2026-08-19
**Universo:** 40 ações do `fixed_40_2018.json`

## Melhor combinação por patrimônio final

- GAP_PERIOD: **75**
- SIGNAL_PERIOD: **9**
- MOMENTUM_PERIOD: **123**
- VOL_PERIOD: **21** (fixo)
- Capital inicial: **R$ 1000.00**
- Capital final: **R$ 30730.01**
- Lucro: **R$ 29730.01**
- Retorno total: **2973.00%**
- CAGR: **48.74%**
- Retorno anual médio: **72.34%**
- Max drawdown: **-56.61%**
- Volatilidade anual: **41.10%**
- Sharpe (rf=0): **1.186**
- Trades: **518**
- Execuções puladas: **0**
- Custos operacionais pagos: **R$ 1243.06**
- Impacto de slippage: **R$ 4460.33**

## Pine original 40/20/63

- Capital final: **R$ 14805.15**
- Retorno total: **1380.51%**
- CAGR: **36.67%**
- Max drawdown: **-63.07%**
- Vantagem patrimonial da vencedora sobre 40/20/63: **107.56%**

## Metodologia

A busca foi exaustiva dentro de GAP 5–80, Signal 2–60 e Momentum 5–252, passo 1. O Vol21 e o Top1 semanal foram preservados para manter a família da estratégia Pine.

A decisão usa apenas o fechamento da última sessão B3 da semana anterior e a execução usa a abertura da primeira sessão da semana seguinte. Não há look-ahead intencional.

**Atenção:** a melhor combinação é in-sample. Ela é a mais lucrativa no histórico testado, não uma promessa de desempenho futuro.
