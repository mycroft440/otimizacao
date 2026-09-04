# Otimização exaustiva 4D — B3 Estratégia Live

**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`
**Combinações testadas:** 65.609.888
**Período:** 2018-01-02 até 2026-09-03
**Grade:** GAP 5–80; Signal 2–60; Momentum 5–252; Vol 2–60.
**Cobertura da grade:** PASS — partições determinísticas, sem sobreposição de GAP e contagem exata por shard.
**Top-100 global:** exato a partir dos Top-100 locais de cada shard.
**Reconciliação motor rápido × replay detalhado:** PASS.

## Melhor combinação

- GAP_PERIOD: **41**
- SIGNAL_PERIOD: **15**
- MOMENTUM_PERIOD: **49**
- VOL_PERIOD: **2**
- Capital inicial: **R$ 1000.00**
- Patrimônio final: **R$ 91231.67**
- Retorno total: **9023.17%**
- CAGR: **68.32%**
- Max drawdown: **-44.33%**
- Sharpe rf=0: **1.4421**
- Trades: **491**
- Execuções puladas: **0**

## Referência 40/20/63/21

- Patrimônio final: **R$ 10791.83**
- Vantagem patrimonial da vencedora: **745.38%**

## Observação

Para evitar materializar mais de 66 milhões de linhas, cada shard testa sua partição inteira e persiste somente seu Top-K. A vencedora e o Top-K global permanecem exatos; o arquivo gigante de todos os resultados deixa de ser necessário.

**Atenção:** a validação estatística continua in-sample; OOS/walk-forward é separada.

## Auditoria endurecida de metricas e valor terminal

- Mark-to-market final: **R$ 91231.67**
- Liquidation equity estimada: **R$ 91110.46**
- Custo estimado para liquidar a posicao terminal: **R$ 121.21**
- Max drawdown reportado: **-44.33%**, medido close-to-close diario.
- Sharpe: **1.4421**, risk-free=0 e anualizacao sqrt(252).
- Anos iniciais e terminais parciais nao entram na media de anos completos.
