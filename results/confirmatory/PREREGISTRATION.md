# Pré-registro da rodada confirmatória (fold 10)

Gravado em 09/nov/2026, **antes** de qualquer leitura do fold 10 pelos métodos abaixo.
Decisão do Eduardo, tomada na mesma data: o material exploratório é suficiente
para o artigo; a rodada confirmatória cobre só os pontos da fronteira de
Pareto (parâmetros × erro). Shapley, redundância, interpretabilidade, curvas
erro × features e heatmap k × n permanecem exploratórios (folds 1–7 / 8 / 9)
e são declarados assim no artigo.

## Protocolo

Protocolo oficial do PTB-XL: treino nos folds 1–8, validação no fold 9 (early
stopping e escolha de época), teste no fold 10. Métrica: AUROC macro nas 5
superclasses. Cada execução é registrada em `results/test_fold_usage.jsonl`.
Nenhuma configuração, época, lista de features ou limiar é escolhida olhando
o fold 10. Todas as execuções listadas são reportadas, inclusive as que
falharem ou piorarem. Uma execução só é repetida se cair antes de ler o fold
10 (por exemplo, falta de memória), e isso também é declarado.

## Execuções congeladas

### 1. Boosting, listas de features fixas

Listas em `results/confirmatory/frozen_feature_lists.json`, com SHA-256:

| conjunto | k | critério de escolha (exploratório) | SHA-256 (16 primeiros) |
|---|---|---|---|
| tsfresh (só série) | 268 | maior AUROC de validação no fold 8 | 3c7c0d8d63fbf093 |
| melhor + multiplex | 537 | maior AUROC de validação no fold 8 | 74b3396bd53d03dc |

- Modelo: `BoostingPorRotulo` (um HistGradientBoosting por rótulo), configuração
  `BOOSTING` de `scripts/evaluate_augmented.py` (learning_rate 0,03,
  max_leaf_nodes 127, min_samples_leaf 80, l2 0, max_features 0,5),
  max_iter 2000, early stopping no fold 9 (n_iter_no_change 30), semente 42.
- Uma execução por conjunto (o boosting é determinístico dada a semente).
- Parâmetros contados como em `scripts/count_boosting_params.py`: limiares +
  valores de folha, soma sobre os 5 classificadores.

### 2. GNN sobre L(HVG)

`scripts/train_line_gnn.py`, configuração idêntica à exploratória
(`results/line_gnn_cv_exploratory/hvg*_d64_l4_e50_test9_val8*`):
hidden 64, 4 camadas, dropout 0,1, 50 épocas, batch 32, AdamW lr 1e-3,
weight decay 1e-2, OneCycle; época escolhida pelo maior AUROC macro no fold 9.

| variante | parâmetros | sementes |
|---|---|---|
| não direcionada | 520.453 | 42, 43, 44, 45, 46 |
| direcionada (`--directed`, 4 relações) | 571.141 | 42, 43, 44, 45, 46 |

Reporta-se média ± desvio das 5 sementes.

### Já executados (não repetidos)

Baselines profundos de configuração fixa, 5 sementes cada, no fold 10
(exceção de 29/out/2026): xresnet1d101, LSTM bidirecional, inception1d,
Transformer, ECG-IMN. 26 usos do fold 10 até este pré-registro.

## Análises previstas sobre os resultados

- Fronteira de Pareto parâmetros × (1 − AUROC), com todos os pontos acima.
- Bootstrap pareado (1.000 reamostragens de registros do fold 10) das
  diferenças: melhor + multiplex − tsfresh; GNN direcionada − xresnet1d101;
  GNN direcionada − GNN não direcionada; melhor + multiplex − LSTM e
  − inception1d (sementes médias nos modelos com várias sementes).
- Nenhuma outra comparação é feita no fold 10 sem novo registro neste arquivo.

## Registro de incidentes

- 09/nov/2026: a fila das GNNs começou com duas execuções simultâneas na GPU
  (RTX 3060, 12 GB). As sementes 42 e 43 da variante direcionada caíram por
  falta de memória de GPU antes da primeira época; a 44 e a 45 foram
  interrompidas ainda carregando o cache. Nenhuma leu o fold 10 (nenhuma
  entrada em `results/test_fold_usage.jsonl`). A fila foi reiniciada com uma
  execução por vez, com as mesmas sementes e a mesma configuração.
- 09/nov/2026: "sementes médias" (seção de análises) era ambíguo. Depois de
  ver os números do boosting, e antes de qualquer resultado da GNN no fold 10,
  fixou-se: a diferença principal usa a AUROC média por semente (como os
  baselines são reportados); a versão com a média dos escores das sementes
  (ensemble, que favorece o lado com mais sementes) é reportada também, em
  `bootstrap_pareado.csv`. As duas aparecem no artigo.
- 12/nov/2026: **rótulos do fold 10 compartilhados**, por decisão do Eduardo,
  num pacote dos três datasets (`export/ecg_fm_bundle/`, gerado por
  `scripts/export_fm_bundle.py`) entregue a outra pessoa para avaliar foundation
  models. O README do pacote fixa o protocolo (treino 1–8, validação 9, teste
  10 lido uma vez por configuração final) e pede as probabilidades por registro
  dos folds 9 e 10. As avaliações feitas por terceiros não passam pelo
  `results/test_fold_usage.jsonl`; o número de leituras do fold 10 informado
  por quem rodar será declarado no artigo junto com os resultados.

## Acréscimo de 13/nov/2026: catch22 + boosting no fold 10

Decisão do Eduardo, para a tabela final: catch22 (264 features, sem seleção)
com o boosting de configuração fixa, treino 1–8, early stopping no fold 9,
fold 10 lido uma vez (`scripts/eval_catch22_fold10.py`). É concorrente sem
seleção; a referência com logística (0,8456) continua reportada.
- 13/nov/2026: sementes 42–46 do ajuste final para catch22, tsfresh e
  melhor + multiplex no fold 10 (listas fixas; ver o pré-registro externo,
  mesma data, e `scripts/boosting_seeds_fold10.py`).
