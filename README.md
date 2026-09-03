# Otimização — B3 Estratégia Live

Este repositório hospeda o workflow e o otimizador da estratégia Pine usada no B3 Strategy Lab. Os dados de mercado, o universo e a proveniência continuam vindo de `mycroft440/b3-strategy-lab`.

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
10. Top 1 apenas. Empate exato preserva a posição incumbente quando ela continua elegível;
11. mesmo Top1 significa manutenção integral, sem nova ordem nem reinvestimento do caixa residual;
12. troca de ativo é atômica: se venda ou nova compra não puder ser executada, a carteira anterior permanece intacta.

O universo é lido de `data/universes/fixed_40_2018.json` e os candles vêm de `data/candles/*_1d.csv` (COTAHIST B3, preços normalizados por splits, sem dividendos/JCP no motor Pine-equivalente).

## Busca padrão

A execução padrão testa **todas as 1.112.032 combinações**:

- `GAP_PERIOD`: 5 a 80, passo 1 (76 valores)
- `SIGNAL_PERIOD`: 2 a 60, passo 1 (59 valores)
- `MOMENTUM_PERIOD`: 5 a 252, passo 1 (248 valores)
- `VOL_PERIOD`: 21 fixo
- rebalanceamento: semanal
- carteira: Top 1

`76 × 59 × 248 = 1.112.032` combinações.

## Configuração canônica e custos

A fonte canônica dos defaults é `optimizer/config.py`:

- capital inicial: R$ 1.000
- custo operacional: **3,25 bps por lado**
- slippage base: 10 bps por lado
- penalidade adicional de lote fracionário: 5 bps ponderada somente sobre as ações fora de lotes completos de 100
- lote mínimo: 1 ação
- dividendos/JCP e IR: não incluídos no modo Pine-equivalente
- alavancagem: 1x

O workflow oficial passa todos esses valores explicitamente e valida capital, custos, datas, períodos e shards antes de iniciar a otimização.

## Endurecimento do backtest

O workflow principal `B3 Pine Live - Otimizacao Exaustiva Hardened` agora exige, antes dos shards:

- dependências instaladas a partir de `optimizer/requirements.txt` com versões fixadas;
- SHA do próprio repositório e SHA do `b3-strategy-lab` congelados;
- testes unitários, regressivos e property tests;
- warm-up calculado a partir dos maiores períodos da grade e auditado no snapshot;
- auditoria de integridade de candles e universo;
- auditoria de gerenciamento de carteira;
- equivalência do precompute acelerado;
- comparação com `optimizer/reference_engine.py`, uma segunda implementação lenta que não reutiliza as rotinas vetorizadas centrais;
- cobertura cartesiana exata das 1.112.032 combinações;
- auditoria financeira de todas as linhas dos shards;
- replay detalhado da vencedora;
- métricas anuais centralizadas e tratamento de primeiro/último ano parcial;
- hashes SHA-256 das fontes, dados e `top_100.csv` no manifesto.

A publicação em `reports/latest/` é recusada se a `main` avançar enquanto o backtest estiver rodando. O resultado não é mais rebased sobre um commit diferente daquele que foi testado.

## Métricas e patrimônio terminal

A camada endurecida distingue **mark-to-market equity** de **liquidation equity estimada**. Drawdown, volatilidade e Sharpe atuais são explicitamente métricas de equity diária close-to-close; o Sharpe usa risk-free zero e anualização `sqrt(252)`. Anos iniciais ou terminais parciais são exibidos, mas não entram na média de anos completos.

## OOS e walk-forward

`validate_top_oos.py` não aceita mais um ranking de treino apenas porque o arquivo foi chamado de "training". Ele exige manifesto `training_only`, `training_end < oos_start` e hash SHA-256 idêntico ao `top_100.csv` utilizado para selecionar parâmetros.

O workflow manual `B3 Pine Live - Walk Forward Exaustivo` implementa janelas rolantes de treino de três anos seguidas do ano OOS seguinte. Cada janela otimiza somente o passado e o capital final OOS é carregado para a janela seguinte.

## Robustez e overfitting

`optimizer/analyze_robustness.py` mede a distribuição de toda a grade, percentis, concentração dos melhores parâmetros e a vizinhança da vencedora. Isso ajuda a detectar um ótimo isolado, mas continua sendo análise in-sample e **não substitui OOS/walk-forward**.

## Universo e survivorship bias

O universo atual é `PINE_EXACT_FIXED_UNIVERSE`: ele existe para reproduzir os 40 slots do Pine e está explicitamente marcado como `survivorship_safe: false`. Isso não deve ser confundido com reconstrução point-in-time do universo histórico da B3.

## Limitação externa ainda aberta

O motor principal passa a ser comparado com uma implementação independente em Python. Ainda assim, a prova mais forte de equivalência com TradingView exige uma fixture externa exportada do Pine contendo sinais, Top1, ordens e equity em datas conhecidas. Até essa fixture existir, o status `pine_external_golden_validation` deve permanecer diferente de `PASS`.

A configuração vencedora in-sample não deve ser interpretada como expectativa de retorno futuro sem OOS/walk-forward e análise de robustez.
