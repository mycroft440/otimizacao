# Fixture externa de referência Pine/TradingView

Esta pasta **não contém uma fixture inventada pelo otimizador**. O golden test só é válido quando os dados são exportados de uma execução externa do script Pine no TradingView e depois congelados aqui.

## Arquivos esperados

Para uma fixture chamada `pine_golden.csv`, deve existir `pine_golden.meta.json` com, no mínimo:

- `source`: exatamente `TradingView Pine external export`;
- `fixture_sha256`: SHA-256 do CSV exportado, calculado antes de qualquer teste Python;
- `pine_script_sha256`: SHA-256 do texto exato do script Pine usado na exportação;
- `exported_at`: timestamp da exportação;
- `start` e `end`;
- `gap_period`, `signal_period`, `momentum_period`, `vol_period`;
- `initial_cash`, `fee_bps`, `slippage_bps`, `odd_lot_extra_bps` quando divergirem do padrão.

O CSV precisa conter, para cada data escolhida como golden:

- `date`;
- `holding`;
- `shares`;
- `cash`;
- `equity`;
- `weekly_target`.

Para uma comparação ainda mais forte, a mesma exportação pode incluir:

- `decision_date`;
- `ticker`;
- `gap_state`;
- `momentum`;
- `vol_valid`;
- `selected_top1`.

## Regra de independência

Não é permitido gerar o CSV com `optimize_b3_pine.py`, `reference_engine.py` ou qualquer outro código deste repositório e depois chamá-lo de golden. Isso apenas criaria uma validação circular.

O auditor `optimizer/audit_pine_golden.py` verifica o hash da fixture, a identificação explícita da fonte externa e compara carteira/equity; quando as colunas de indicador existem, também compara sinais e seleção de Top1.

Enquanto uma fixture externa real não estiver versionada e não passar no auditor, `VALIDATION_STATUS.json` deve continuar reportando `pine_external_golden_validation` como pendente, nunca como `PASS`.
