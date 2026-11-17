# InterpretECG

Experimentos de múltiplas visões e interpretabilidade no PTB-XL. O protocolo
completo está descrito em `CLAUDE.md`.

## Ambiente

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Baseline Catch22

O baseline extrai as 22 features Catch22 de cada uma das 12 derivações
(264 atributos), usa os folds oficiais 1–8 para treino, o fold 9 para selecionar
o `C` e os limiares, e avalia uma única vez no fold 10.

```bash
source .venv/bin/activate
catch22-baseline
```

O comando recusa executar a avaliação final enquanto algum sinal estiver
ausente ou incompleto. A extração é incremental: as features são preservadas em
`data/features/catch22_12leads.parquet` e uma execução interrompida pode ser
retomada com o mesmo comando.

Saídas finais em `results/catch22_baseline/`:

- `metrics.json`: AUROC, average precision e métricas com limiar;
- `model.joblib`: pipeline treinado, labels, limiares e ordem das features;
- `test_predictions.parquet`: probabilidades e decisões por ECG no fold 10.

Para um teste parcial explicitamente não científico:

```bash
catch22-baseline --allow-incomplete --max-records-per-fold 100
```

Para apenas adiantar o cache enquanto o download está em andamento:

```bash
catch22-baseline --allow-incomplete --extract-only
```

## Visões derivadas (imagem e grafo)

Os operadores que re-representam uma derivação como imagem ou como grafo estão
em `src/interpretecg/views/`, copiados do projeto irmão CNA. A procedência, as
alterações feitas na cópia e duas divergências herdadas estão em
[`src/interpretecg/views/PROVENANCE.md`](src/interpretecg/views/PROVENANCE.md).

```python
from interpretecg.views.series_to_image import (
    spectrogram, scalogram, gramian_angular_field,
    markov_transition_field, recurrence_plot,
)
from interpretecg.views.series_to_graph import (
    visibility_graph, transition_graph, takens_recurrence_graph,
    extract_features, features_to_array,
)
from interpretecg.views.image_to_tabular import haralick_features, hog_features
```

### Extração

```bash
extract-views --n-jobs 96                 # 10 famílias, 12 derivações
python scripts/normalize_view_cache.py    # dropa as colunas constantes
```

A extração é incremental e cacheada por família em `data/features/views/`, um
Parquet mais um manifesto por família. Interromper e repetir o comando retoma de
onde parou; um checkpoint é gravado a cada `--checkpoint-every` registros.

O embedding de Takens é congelado no nível do dataset, em
`data/features/takens_embedding.json`, estimado nos folds de treino. A imagem de
recorrência e a rede de Takens usam os **mesmos** parâmetros de propósito, para
que a seção 4.4 do protocolo compare duas leituras de uma única matriz.

`normalize_view_cache.py` remove as colunas que são constantes por construção
(`n_components` e `largest_cc_frac` nos grafos de visibilidade, onde a
conectividade é garantida). Isso importa porque o controle placebo casa a
dimensionalidade da família, e uma largura inflada tornaria o controle inválido.
O script se recusa a dropar qualquer coluna que não seja de fato constante, e é
idempotente.

Todos os operadores são transformações determinísticas e sem treino de uma série 1D. O
`extract_features(G, kind=...)` entrega 26 descritores comuns mais um escalar
diagnóstico específico da conversão (expoente de grau para visibilidade, taxa
de entropia para transição, dimensão de correlação para Takens), de propósito
com a mesma dimensionalidade entre conversões.

Custo medido numa derivação de 1.000 amostras (100 Hz): as cinco conversões
para imagem somam ~45 ms; o gargalo está no readout de Takens (~2,2 s) e na
construção do grafo de visibilidade natural (~1,5 s). Ver
`PROVENANCE.md` para o detalhamento.

Testes:

```bash
.venv/bin/python -m pytest tests/views -q
```

## Regimes de avaliação e o fold selado

O fold 10 fica **fora** de toda análise exploratória. Cada comparação avaliada
nele seria um uso do conjunto de teste, e o protocolo da seção 6 do `CLAUDE.md`
prevê dezenas delas (visões isoladas, combinações, placebo, nulls de
reparametrização). Sem essa separação, o número final deixaria de ser uma
estimativa honesta de generalização.

| regime | treino | validação | teste |
|---|---|---|---|
| `exploratory` (padrão) | folds 1–7 | fold 8 | fold 9 |
| `confirmatory` | folds 1–8 | fold 9 | **fold 10** |

O regime confirmatório é o protocolo oficial do PTB-XL, o mesmo que Strodthoff
et al. reportam, então é dele que sai o número comparável à literatura.

```bash
catch22-baseline                                    # exploratório, o padrão
catch22-baseline --regime confirmatory \
    --confirm-test-fold "baseline final do artigo"  # exige justificativa
```

O regime confirmatório recusa rodar sem `--confirm-test-fold`, e cada execução
é registrada em `results/test_fold_usage.jsonl` com data, justificativa e
resultado, para que a contagem de avaliações no fold 10 seja auditável.

O cache de features não precisa ser separado por regime: as transformações são
determinísticas e nunca recebem rótulos, então extrair features do fold 10 não
é vazamento — vazamento seria *avaliar* nele.

## Registro de experimentos

Todo resultado avaliado entra num log append-only, `results/experiments.jsonl`,
que é a fonte da verdade. A tabela legível `results/RESULTS.md` é **sempre
regenerada** a partir dele e não deve ser editada à mão.

```bash
python scripts/log_experiment.py --metrics <caminho/metrics.json> \
    --label <id-curto> --view <series|image|graph> --family <nome> \
    --question "<o que a execucao pretendia responder>" \
    --finding "<o que os numeros mostraram>"

python scripts/log_experiment.py --rerender    # so regenera a tabela
```

Em sessões com Claude Code o mesmo fluxo está disponível como `/log`, que lê o
`metrics.json` mais recente e preenche automaticamente o que já está lá —
regime, folds, classificador, contagem de features, métricas.

O script registra uma impressão digital SHA-256 do código em `src/`, para que
qualquer número no log possa ser rastreado até a versão que o produziu. Entradas
corrigidas por experimentos posteriores carregam um bloco de correção e
aparecem marcadas com ⚠️ na tabela: o achado errado é preservado, não apagado.
