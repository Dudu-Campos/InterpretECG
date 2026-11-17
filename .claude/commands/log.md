---
description: Registra um resultado de experimento no log append-only do projeto e regenera a tabela de resultados.
---

Registre o experimento indicado em `$ARGUMENTS` no log do projeto.

Sem argumentos, registre a execução mais recente: encontre o `metrics.json`
mais novo sob `results/`.

## Procedimento

1. Leia o `metrics.json` da execução. Dele saem automaticamente o regime, os
   folds, o classificador, a contagem de features, os tamanhos dos splits, o
   `C` selecionado e todas as métricas — não os redigite.
2. Confira se esse `metrics.json` já está em `results/experiments.jsonl`
   (campo `artifacts.metrics`). Se estiver, **não** duplique: ou atualize a
   entrada existente, ou avise que já foi registrado.
3. Preencha os campos que o `metrics.json` não contém, porque são justamente
   os que dão sentido ao número:
   - `--label`: identificador curto e estável, ex. `catch22-12leads`.
   - `--view`: `series`, `image`, `graph` ou a combinação avaliada.
   - `--family`: uma por família de features (repetível), ex. `--family catch22`.
   - `--leads`: as derivações em que as visões foram calculadas.
   - `--params`: objeto JSON com os parâmetros dos operadores que importam
     para reproduzir (ρ do LPHVG, taxa de recorrência, `m` e `τ` do embedding,
     `n_bins`, tamanho de imagem).
   - `--question`: o que esta execução pretendia responder. Uma frase.
   - `--finding`: o que os números mostraram. Uma ou duas frases, concretas.
     Se o achado contradiz uma hipótese do `CLAUDE.md`, diga isso.
4. Execute:

```bash
.venv/bin/python scripts/log_experiment.py --metrics <caminho> --label <...> \
    --view <...> --family <...> --leads <...> --params '<json>' \
    --question "<...>" --finding "<...>"
```

5. Relate em uma linha o que foi registrado e o total de experimentos no log.

## Regras

- `results/experiments.jsonl` é **append-only** e é a fonte da verdade.
  `results/RESULTS.md` é sempre regenerado a partir dele; nunca o edite à mão.
- O script registra uma impressão digital SHA-256 do código em `src/`, para
  que qualquer número no log possa ser rastreado até a versão que o produziu.
- Não invente `--finding`. Se os números não sustentam uma conclusão, escreva
  que não sustentam. Um log com achados inflados é pior que um log vazio.
- Execuções no regime `confirmatory` também já aparecem em
  `results/test_fold_usage.jsonl`; são registros distintos e ambos devem
  existir.
- Para só regenerar a tabela sem registrar nada: `--rerender`.
