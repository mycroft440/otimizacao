# Otimização — B3 Estratégia Live

Este repositório hospeda **somente o workflow e o otimizador** da estratégia Pine usada no B3 Strategy Lab.
Os dados de mercado, o universo e a proveniência continuam vindo de `mycroft440/b3-strategy-lab`.

## Estratégia reproduzida

A busca preserva a mesma família do Pine **Gap Momentum + Momentum positivo + Top 1 semanal**:

1. `gap = open - close[1]`;
2. soma dos gaps positivos e negativos em `GAP_PERIOD`;
3. `Gap Ratio = 100 * positiveSum / negativeSum` (quando `negativeSum == 0`, usa `1.0`, como no Pine);
4. SMA do Gap Ratio em `SIGNAL_PERIOD`;
5. estado comprado quando a SMA sobe e vendido quando cai, persistindo quando fica igual;
6. `Momentum = close / close[MOMENTUM_PERIOD] - 1`;
7. ação elegível somente com estado Gap comprado, momentum positivo e volatilidade amostral de 21 retornos positiva;
8. no último fechamento da semana é escolhido o maior Momentum entre as ações elegíveis;
9. execução na abertura da primeira sessão B3 da semana seguinte;
10. Top 1 apenas. Empate exato preserva a posição incumbente quando ela continua elegível.

O universo é lido de `data/universes/fixed_40_2018.json` do B3 Strategy Lab e os candles vêm de `data/candles/*_1d.csv` (COTAHIST B3, preços normalizados por splits, sem dividendos/JCP no motor Pine-equivalente).

## Busca padrão

A execução padrão testa **todas as 1.112.032 combinações** dentro do domínio:

- `GAP_PERIOD`: 5 a 80, passo 1 (76 valores)
- `SIGNAL_PERIOD`: 2 a 60, passo 1 (59 valores)
- `MOMENTUM_PERIOD`: 5 a 252, passo 1 (248 valores)
- `VOL_PERIOD`: 21 fixo, para manter a semântica do Pine em que volatilidade é apenas um gate `> 0`
- rebalanceamento: semanal fixo
- carteira: Top 1 fixo

`76 × 59 × 248 = 1.112.032` combinações.

O workflow divide a busca em shards paralelos. Cada shard testa integralmente seu subconjunto; o job final junta os CSVs, ordena por patrimônio final e recalcula a curva diária detalhada das melhores configurações.

## Custos padrão

Para espelhar o Pine auditado:

- capital inicial: R$ 1.000
- custo operacional: 3 bps por lado
- slippage base: 10 bps por lado
- penalidade adicional de lote fracionário: 5 bps ponderada somente sobre as ações fora de lotes completos de 100
- lote mínimo: 1 ação
- dividendos/JCP e IR: não incluídos
- alavancagem: 1x nesta otimização de parâmetros

Os valores podem ser alterados no disparo manual do workflow.

## Resultados

O workflow publica:

- `all_results.csv.gz`: todas as combinações testadas;
- `top_100.csv`: as 100 mais lucrativas;
- `BEST.json`: melhor configuração e métricas;
- `BEST_EQUITY_DAILY.csv`: curva diária da vencedora;
- `OPTIMIZATION_SUMMARY.md`: resumo humano;
- `MANIFEST.json`: commit do B3 Strategy Lab, domínio de busca, custos, contagem esperada/testada e hashes SHA-256.

O job final também grava o resumo no GitHub Actions Job Summary. Quando o token do workflow possui permissão de escrita, `reports/latest/` é atualizado na `main` com commit `[skip ci]`.

## Observação metodológica

A configuração vencedora é **in-sample**: escolher o melhor parâmetro usando o mesmo período em que seu desempenho é medido produz viés de otimização. O workflow mostra a mais lucrativa no histórico solicitado, mas não rotula esse número como desempenho futuro ou out-of-sample.
