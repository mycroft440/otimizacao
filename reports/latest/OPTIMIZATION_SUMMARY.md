# Otimização exaustiva 4D — B3 Estratégia Live

**Status:** `IN_SAMPLE_EXHAUSTIVE_SUCCESS`
**Combinações lógicas cobertas:** 65.609.888
**Simulações físicas executadas:** 8.896.256
**Redução por equivalência exata de VOL:** 7,375x
**VOL gates distintos:** 8 para os 59 valores VOL 2–60
**Paralelismo:** 20 shards independentes no GitHub Actions
**Período:** 2018-01-02 até 2026-09-02
**Grade:** GAP 5–80; Signal 2–60; Momentum 5–252; Vol 2–60.
**Cobertura da grade:** PASS — partições determinísticas, sem sobreposição de GAP e contagem exata por shard.
**Top-100 global:** exato a partir dos Top-100 locais de cada shard.
**Reconciliação motor rápido × replay detalhado:** PASS.

A expressão anterior "65.609.888 combinações testadas" era ambígua. A busca cobre logicamente todas as 65.609.888 combinações de parâmetros, mas não executa uma carteira independente para cada uma delas. Como o Pine usa VOL apenas como gate booleano `volatilidade > 0` com os thresholds padrão, os 59 períodos de VOL produziram somente 8 matrizes de elegibilidade distintas no snapshot deste run. Matrizes booleanas byte-a-byte idênticas são simuladas uma única vez e o resultado é reutilizado somente para os VOLs comprovadamente equivalentes.

Grupos de VOL equivalentes encontrados no snapshot do run #33: **[2]**, **[3–32]**, **[33–36]**, **[37–41]**, **[42–46]**, **[47–50]**, **[51–55]**, **[56–60]**.

## Melhor combinação

- GAP_PERIOD: **41**
- SIGNAL_PERIOD: **15**
- MOMENTUM_PERIOD: **49**
- VOL_PERIOD: **2**
- Capital inicial: **R$ 1000.00**
- Patrimônio final: **R$ 89280.33**
- Retorno total: **8828.03%**
- CAGR: **67.93%**
- Max drawdown: **-44.33%**
- Sharpe rf=0: **1.4367**
- Trades: **491**
- Execuções puladas: **0**

## Referência 40/20/63/21

- Patrimônio final: **R$ 10561.35**
- Vantagem patrimonial da vencedora: **745.35%**

## Observação

O motor evita dois custos diferentes: não materializa dezenas de milhões de linhas e também reutiliza simulações quando a entrada efetiva de VOL é exatamente a mesma matriz booleana. Essa reutilização é uma memoização exata, não uma aproximação, poda heurística ou amostragem da grade.

No shard 0, por exemplo, 3.453.152 combinações lógicas corresponderam a 468.224 simulações físicas. A etapa pesada desse shard foi de aproximadamente 20:34:58 a 20:36:47 UTC no run #33, cerca de 1m49s, usando NumPy vetorizado. Os 20 shards rodaram em paralelo. O workflow completo começou às 20:28:20 UTC e terminou às 20:38:18 UTC.

**Atenção:** a validação estatística continua in-sample; OOS/walk-forward é separada.

## Auditoria endurecida de metricas e valor terminal

- Mark-to-market final: **R$ 89280.33**
- Liquidation equity estimada: **R$ 89161.71**
- Custo estimado para liquidar a posicao terminal: **R$ 118.62**
- Max drawdown reportado: **-44.33%**, medido close-to-close diario.
- Sharpe: **1.4367**, risk-free=0 e anualizacao sqrt(252).
- Anos iniciais e terminais parciais nao entram na media de anos completos.