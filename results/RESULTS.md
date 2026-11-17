# Registro de experimentos

Gerado por `scripts/log_experiment.py` a partir de
`results/experiments.jsonl`. **Não editar à mão**: as alterações são
sobrescritas na próxima execução de `/log`.

59 experimento(s) registrado(s).

| # | data | rótulo | visão | features | regime | teste | AUROC macro | macro-AP |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-26 | catch22-12leads | series | 264 | confirmatory | fold 10 | 0.8456 | 0.6606 | ⚠️
| 2 | 2026-10-26 | catch22-12leads-exploratory | series | 264 | exploratory | fold 9 | 0.8425 | 0.6395 |
| 3 | 2026-10-27 | catch22-12leads-gbm | series | 264 | exploratory | fold 9 | 0.8726 | 0.7025 |
| 4 | 2026-10-27 | all-views-12leads | series+image+graph | 3198 | exploratory | fold 9 | 0.8969 | 0.7463 |
| 5 | 2026-10-27 | image-view-12leads | image | 1776 | exploratory | fold 9 | 0.8781 | 0.7011 |
| 6 | 2026-10-27 | graph-view-12leads | graph | 1158 | exploratory | fold 9 | 0.8733 | 0.7103 |
| 7 | 2026-10-27 | all-views-12leads-mlp | series+image+graph | 3198 | exploratory | fold 9 | 0.9030 | 0.7656 | ⚠️
| 8 | 2026-10-27 | image-graph-12leads-mlp | image+graph | 2934 | exploratory | fold 9 | 0.9000 | 0.7622 | ⚠️
| 9 | 2026-10-27 | graph-view-12leads-mlp | graph | 1158 | exploratory | fold 9 | 0.8874 | 0.7408 | ⚠️
| 10 | 2026-10-27 | catch22-12leads-mlp | series | 264 | exploratory | fold 9 | 0.8697 | 0.6925 |
| 11 | 2026-10-29 | cnn-xresnet1d101-exploratory | series | — | exploratory | fold 9 | 0.9287 | 0.8132 |
| 12 | 2026-10-29 | cnn-xresnet1d101 | series | — | confirmatory | fold 10 | 0.9304 | 0.8270 |
| 13 | 2026-10-30 | all-views-12leads-widegrid | series+image+graph | 3198 | exploratory | fold 9 | 0.9007 | 0.7563 |
| 14 | 2026-10-30 | catch22-12leads-widegrid | series | 264 | exploratory | fold 9 | 0.8768 | 0.7090 |
| 15 | 2026-10-30 | graph-view-12leads-widegrid | graph | 1158 | exploratory | fold 9 | 0.8794 | 0.7186 |
| 16 | 2026-10-30 | image-view-12leads-widegrid | image | 1776 | exploratory | fold 9 | 0.8796 | 0.7199 |
| 17 | 2026-10-30 | image-graph-12leads-widegrid | image+graph | 2934 | exploratory | fold 9 | 0.8925 | 0.7410 |
| 18 | 2026-11-03 | series-amplitude-12leads | series | 60 | exploratory | fold 9 | 0.8568 | 0.6785 |
| 19 | 2026-11-03 | series-interlead-12leads | series | 39 | exploratory | fold 9 | 0.7908 | 0.5837 |
| 20 | 2026-11-03 | series-placebo-12leads | series | 99 | exploratory | fold 9 | 0.4964 | 0.2581 |
| 21 | 2026-11-03 | catch22-amplitude-12leads | series | 324 | exploratory | fold 9 | 0.9026 | 0.7632 |
| 22 | 2026-11-03 | catch22-interlead-12leads | series | 303 | exploratory | fold 9 | 0.8844 | 0.7239 |
| 23 | 2026-11-03 | catch22-amplitude-interlead-12leads | series | 363 | exploratory | fold 9 | 0.9053 | 0.7703 |
| 24 | 2026-11-03 | catch22-placebo-12leads | series | 363 | exploratory | fold 9 | 0.8741 | 0.7046 |
| 25 | 2026-11-03 | all-views-amplitude-interlead-12leads | series+image+graph | 3297 | exploratory | fold 9 | 0.9147 | 0.7880 |
| 26 | 2026-11-03 | all-views-placebo-12leads | series+image+graph | 3297 | exploratory | fold 9 | 0.9022 | 0.7562 |
| 27 | 2026-11-03 | catch22-amplitude-12leads-mlp | series | 324 | exploratory | fold 9 | 0.8934 | 0.7480 |
| 28 | 2026-11-03 | catch22-interlead-12leads-mlp | series | 303 | exploratory | fold 9 | 0.8768 | 0.7141 |
| 29 | 2026-11-03 | catch22-amplitude-interlead-12leads-mlp | series | 363 | exploratory | fold 9 | 0.8958 | 0.7582 |
| 30 | 2026-11-03 | catch22-placebo-12leads-mlp | series | 363 | exploratory | fold 9 | 0.8655 | 0.6864 |
| 31 | 2026-11-03 | all-views-amplitude-interlead-12leads-mlp | series+image+graph | 3297 | exploratory | fold 9 | 0.9089 | 0.7774 |
| 32 | 2026-11-03 | all-views-placebo-12leads-mlp | series+image+graph | 3297 | exploratory | fold 9 | 0.9021 | 0.7640 |
| 33 | 2026-11-06 | image-spectrogram-abs-12leads | image | 336 | exploratory | fold 9 | 0.8493 | 0.6442 |
| 34 | 2026-11-06 | image-scalogram-abs-12leads | image | 336 | exploratory | fold 9 | 0.8650 | 0.6837 |
| 35 | 2026-11-06 | image-gasf-abs-12leads | image | 336 | exploratory | fold 9 | 0.8700 | 0.7108 |
| 36 | 2026-11-06 | image-gadf-abs-12leads | image | 336 | exploratory | fold 9 | 0.8573 | 0.6565 |
| 37 | 2026-11-06 | image-mtf-abs-12leads | image | 336 | exploratory | fold 9 | 0.7657 | 0.5149 |
| 38 | 2026-11-06 | image-recurrence-abs-12leads | image | 108 | exploratory | fold 9 | 0.8184 | 0.5975 |
| 39 | 2026-11-06 | graph-takens-abs-12leads | graph | 300 | exploratory | fold 9 | 0.8112 | 0.5821 |
| 40 | 2026-11-06 | graph-transition-abs-12leads | graph | 324 | exploratory | fold 9 | 0.7976 | 0.5769 |
| 41 | 2026-11-06 | image-view-abs-12leads | image | 1788 | exploratory | fold 9 | 0.8924 | 0.7521 |
| 42 | 2026-11-06 | graph-view-abs-12leads | graph | 1176 | exploratory | fold 9 | 0.8780 | 0.7115 |
| 43 | 2026-11-06 | catch22-views-abs-12leads | series+image+graph | 3228 | exploratory | fold 9 | 0.9106 | 0.7799 |
| 44 | 2026-11-06 | catch22-amplitude-interlead-views-abs-12leads | series+image+graph | 3327 | exploratory | fold 9 | 0.9113 | 0.7833 |
| 45 | 2026-11-06 | catch22-amplitude-interlead-views-abs-norec-12leads | series+image+graph | 2919 | exploratory | fold 9 | 0.9128 | 0.7847 |
| 46 | 2026-11-06 | image-view-abs-12leads-mlp | image | 1788 | exploratory | fold 9 | 0.9014 | 0.7685 |
| 47 | 2026-11-06 | graph-view-abs-12leads-mlp | graph | 1176 | exploratory | fold 9 | 0.8850 | 0.7300 |
| 48 | 2026-11-06 | catch22-views-abs-12leads-mlp | series+image+graph | 3228 | exploratory | fold 9 | 0.9089 | 0.7817 |
| 49 | 2026-11-06 | catch22-amplitude-interlead-views-abs-12leads-mlp | series+image+graph | 3327 | exploratory | fold 9 | 0.9109 | 0.7861 |
| 50 | 2026-11-06 | catch22-amplitude-interlead-views-abs-norec-12leads-mlp | series+image+graph | 2919 | exploratory | fold 9 | 0.9123 | 0.7893 |
| 51 | 2026-11-09 | multiplex-hvg-lphvg-8leads | graph | 196 | exploratory | fold 9 | 0.8465 | 0.6522 |
| 52 | 2026-11-09 | catch22-amplitude-multiplex | series+graph | 520 | exploratory | fold 9 | 0.9085 | 0.7759 |
| 53 | 2026-11-09 | base-multiplex | series+graph | 559 | exploratory | fold 9 | 0.9097 | 0.7792 |
| 54 | 2026-11-09 | base-directed-multiplex | series+graph | 1271 | exploratory | fold 9 | 0.9170 | 0.7943 |
| 55 | 2026-11-09 | best-set-multiplex | series+graph | 2952 | exploratory | fold 9 | 0.9210 | 0.8030 |
| 56 | 2026-11-09 | lstm-bidir-confirmatory | series | — | confirmatory | fold 10 | 0.9200 | 0.8074 |
| 57 | 2026-11-09 | inception1d-confirmatory | series | — | confirmatory | fold 10 | 0.9197 | 0.8136 |
| 58 | 2026-11-09 | transformer-deepecgkit-confirmatory | series | — | confirmatory | fold 10 | 0.9025 | 0.7698 |
| 59 | 2026-11-09 | xresnet1d101-confirmatory-5seeds | series | — | confirmatory | fold 10 | 0.9286 | 0.8223 |

## 1. catch22-12leads

**Pergunta:** Qual o patamar da visao serie com features genericas de serie temporal, antes de qualquer visao derivada?

**Achado:** AUROC macro 0,846 no fold 10. Fica 0,028 ABAIXO do baseline handcrafted Wavelet+NN do benchmark (0,874) e 0,084 abaixo do melhor CNN (resnet1d_wang, 0,930), que corta o erro de ranqueamento em 55%. A grade de C varia so na quarta decimal em quatro ordens de magnitude: o gargalo e a representacao, nao o classificador.

> **Corrigido por `catch22-12leads-gbm`.** O achado concluia 'o gargalo e a representacao, nao o classificador' a partir da grade de C chapada. A grade de C varia apenas a regularizacao DENTRO da familia linear, logo nao separa limitacao de representacao de limitacao de capacidade. Gradient boosting nas mesmas features ganha +0,0301 de AUROC.

> O que permanece: A parte estreita do achado continua valida: com 264 features e 15 mil amostras, o modelo nao esta limitado por overfitting.

- Regime `confirmatory`: treino [1, 2, 3, 4, 5, 6, 7, 8], validação fold 9, teste fold 10
- Famílias: catch22
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"catch24": false, "features_per_lead": 22}`
- `C` selecionado na validação: 0.1
- AUROC por classe: NORM 0.8708, MI 0.8170, STTC 0.8761, CD 0.8559, HYP 0.8080
- Código: src 2a03e4a60c15ddb7
- Métricas completas: `/exp/duducampos/InterpretECG/results/catch22_baseline/metrics.json`

## 2. catch22-12leads-exploratory

**Pergunta:** O holdout exploratorio (fold 9) reproduz a ordenacao do fold 10, de modo que o fold 10 possa ficar selado?

**Achado:** AUROC macro 0,8425 no fold 9 contra 0,8456 no fold 10, diferenca de 0,003 com 12,5% menos dados de treino. Serve para ordenar comparacoes agregadas. Por classe o desvio e maior (HYP -0,040, MI +0,025), entao afirmacoes por classe tiradas da exploracao exigem cautela.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"catch24": false, "features_per_lead": 22}`
- `C` selecionado na validação: 10.0
- AUROC por classe: NORM 0.8778, MI 0.8415, STTC 0.8660, CD 0.8586, HYP 0.7684
- Código: src 2a03e4a60c15ddb7
- Métricas completas: `/exp/duducampos/InterpretECG/results/catch22_baseline_exploratory/metrics.json`

## 3. catch22-12leads-gbm

**Pergunta:** A limitacao do Catch22 vem da representacao ou da classe de modelo? A grade de C so varia regularizacao DENTRO da familia linear, entao nao separa as duas coisas.

**Achado:** CORRIGE o achado da entrada 1. Trocando so a classe de modelo nas mesmas 264 features: logistica 0,8425 -> random forest 0,8613 -> gradient boosting 0,8726 no fold 9. Ganho de +0,0301, ou 19% menos erro de ranqueamento. Mais de um terco do gap para a CNN era capacidade de classificador, nao acessibilidade de padrao. O par lr=0,05/leaves=63 ainda supera lr=0,1/leaves=31, entao 0,8726 provavelmente nao e o teto.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"classifier": "HistGradientBoosting", "early_stopping": true, "learning_rate": 0.05, "max_iter": 400, "max_leaf_nodes": 63}`
- AUROC por classe: NORM 0.9020, MI 0.8832, STTC 0.8951, CD 0.8945, HYP 0.7882
- Código: src 2ccd414cf52c33e9
- Métricas completas: `/exp/duducampos/InterpretECG/results/catch22_gbm_exploratory/metrics.json`

## 4. all-views-12leads

**Pergunta:** Combinar as visoes de imagem e grafo ao catch22 acrescenta desempenho, e o ganho vem de visoes distintas ou de reparametrizacoes de uma mesma visao?

**Achado:** AUROC macro 0,8969 no fold 9 contra 0,8715 do catch22 com o mesmo boosting (+0,025). Na curva cumulativa (results/cumulative_curve) a LPHVG sozinha responde por +0,017 [0,012; 0,022] e a curva fica plana a partir de cinco familias; HVG depois de LPHVG (+0,001) e GASF depois de GADF (+0,0004) tem IC contendo zero, como preve o null de reparametrizacao. NAO controla dimensionalidade (3198 contra 264 features): falta o placebo da secao 6.3. Ganho nao-linear cai de +0,032 no catch22 para +0,008, mas as grades de C e de boosting sao estreitas e C=0,01 bateu no piso.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "graph_readout": "26 descritores + 1 escalar por conversao; n_components/largest_cc_frac descartados nos visibility graphs", "image_readout": "Haralick 14x(media,amplitude); RP via RQA (8)", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9186, MI 0.9042, STTC 0.9100, CD 0.9030, HYP 0.8485
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_exploratory/todas_as_visoes.json`

## 5. image-view-12leads

**Pergunta:** A visao de imagem sozinha (seis familias) alcanca o catch22, e com qual classe de modelo?

**Achado:** Logistica C=0,01 vence o boosting na validacao (0,8917 contra 0,8852) e da 0,8781 no fold 9, acima do catch22 com boosting (0,8715). Sugere que a imagem entrega linearmente separavel o que o catch22 so entrega a modelo nao-linear, mas C=0,01 esta no piso da grade e o boosting teve so duas configuracoes. HYP 0,834 contra 0,793 do catch22; afirmacao por classe exige bootstrap pareado.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "graph_readout": "26 descritores + 1 escalar por conversao; n_components/largest_cc_frac descartados nos visibility graphs", "image_readout": "Haralick 14x(media,amplitude); RP via RQA (8)", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9090, MI 0.8723, STTC 0.8898, CD 0.8852, HYP 0.8341
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_exploratory/visao_imagem.json`

## 6. graph-view-12leads

**Pergunta:** A visao de grafo sozinha (quatro familias) alcanca o catch22, e onde ela se destaca por classe?

**Achado:** AUROC macro 0,8733 no fold 9 com boosting, empatada com o catch22 (0,8715); ganho nao-linear de so +0,002 na validacao. Por classe STTC 0,906 (catch22 0,896) e HYP 0,793 (igual ao catch22), ao contrario da imagem que sobe HYP. Indicio para H2, nao medicao: falta bootstrap pareado por classe.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "graph_readout": "26 descritores + 1 escalar por conversao; n_components/largest_cc_frac descartados nos visibility graphs", "image_readout": "Haralick 14x(media,amplitude); RP via RQA (8)", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9092, MI 0.8819, STTC 0.9057, CD 0.8769, HYP 0.7926
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_exploratory/visao_grafo.json`

## 7. all-views-12leads-mlp

**Pergunta:** Um MLP encontra nas features combinadas relacoes nao-lineares que logistica e boosting nao pegam?

**Achado:** MLP 0,9030 (ensemble de 5 sementes) e 0,8993 +- 0,0013 por semente, contra 0,8969 do boosting: +0,002 por semente, provavelmente dentro do ruido (sem bootstrap pareado). Ganho das visoes sobre o catch22 sob MLP e +0,033 (media por semente contra media por semente), contra +0,025 sob boosting. Rasos tiveram grade menor que o MLP (2 configs de boosting contra 6).

> **Corrigido por `all-views-12leads-widegrid`.** A comparacao era contra boosting com so 2 configuracoes e early stopping em 10% do treino. Com a grade ampla o boosting chega a 0,9007 e o MLP por semente fica 0,001 ABAIXO, nao +0,002 acima.

> O que permanece: O ganho das visoes sobre o catch22 depende do leitor: +0,033 sob MLP por semente contra +0,024 sob boosting ajustado.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "lphvg_rho": 1, "mlp": "6 configs, AdamW lr 3e-4, batch 256, early stopping no fold 8 (paciencia 20), clip +-5 sd, ensemble de 5 sementes", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9306, MI 0.9036, STTC 0.9180, CD 0.9066, HYP 0.8562
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/todas_as_visoes.json`

## 8. image-graph-12leads-mlp

**Pergunta:** Sem o catch22, as visoes derivadas ganham com um leitor denso nao-linear?

**Achado:** MLP 0,9000 (ensemble) e 0,8966 +- 0,0011 por semente, contra 0,8898 do boosting: +0,007 por semente. Ressalva: grades dos rasos estreitas; falta bootstrap pareado.

> **Corrigido por `image-graph-12leads-widegrid`.** A vantagem do MLP de +0,007 era contra a grade antiga; contra o boosting ajustado e +0,004.

> O que permanece: Sem o catch22 o MLP continua a frente do boosting ajustado, mas falta bootstrap pareado.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "lphvg_rho": 1, "mlp": "6 configs, AdamW lr 3e-4, batch 256, early stopping no fold 8 (paciencia 20), clip +-5 sd, ensemble de 5 sementes", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9283, MI 0.8938, STTC 0.9155, CD 0.9070, HYP 0.8553
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/imagem_grafo.json`

## 9. graph-view-12leads-mlp

**Pergunta:** A visao de grafo, que empatava com o catch22 sob boosting, tem informacao que so um leitor denso acessa?

**Achado:** MLP 0,8874 (ensemble) e 0,8839 +- 0,0007 por semente, contra 0,8733 do boosting (+0,011) e contra 0,8660 do catch22 sob o mesmo MLP. Sob boosting grafo e catch22 empatavam; sob MLP o grafo fica 0,018 acima. Sugere informacao codificada em combinacao de features, nao em cortes por eixo; depende de as grades dos rasos nao estarem subajustadas.

> **Corrigido por `graph-view-12leads-widegrid`.** Dizia +0,011 do MLP sobre o boosting e que grafo e catch22 empatavam sob boosting. Com a grade ampla a vantagem do MLP cai para +0,004 e o grafo supera o catch22 em +0,003 sob boosting.

> O que permanece: A distancia grafo - catch22 depende do leitor: +0,003 sob boosting ajustado, +0,018 sob MLP.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "lphvg_rho": 1, "mlp": "6 configs, AdamW lr 3e-4, batch 256, early stopping no fold 8 (paciencia 20), clip +-5 sd, ensemble de 5 sementes", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9223, MI 0.8898, STTC 0.9145, CD 0.8909, HYP 0.8195
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/visao_grafo.json`

## 10. catch22-12leads-mlp

**Pergunta:** O MLP acrescenta sobre o boosting no catch22?

**Achado:** Nao. MLP 0,8697 (ensemble) e 0,8660 +- 0,0025 por semente, contra 0,8715 do boosting (-0,006 por semente). Unico dos 17 conjuntos em que o MLP perde: features do catch22 sao escolhidas para serem individualmente informativas, o que favorece arvores.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "lphvg_rho": 1, "mlp": "6 configs, AdamW lr 3e-4, batch 256, early stopping no fold 8 (paciencia 20), clip +-5 sd, ensemble de 5 sementes", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9024, MI 0.8793, STTC 0.8932, CD 0.8749, HYP 0.7989
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22.json`

## 11. cnn-xresnet1d101-exploratory

**Pergunta:** Qual a distancia real entre as visoes handcrafted e a CNN de referencia, medida no mesmo protocolo exploratorio?

**Achado:** AUROC macro 0,9287 no fold 9, igual ao publicado para a xresnet1d101 (0,928 no fold 10), o que valida a reimplementacao. Distancia para o melhor conjunto de visoes no mesmo fold: 0,026 (MLP, ensemble) a 0,032 (boosting da grade antiga). Por classe a maior distancia e HYP (0,903 contra 0,849) e a menor STTC (0,926 contra 0,910).

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: raw_signal
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"agregacao": "max", "arquitetura": "xresnet1d101", "early_stopping": null, "epocas": 50, "janela": 250, "lote": 128, "lr_max": 0.01, "passo_avaliacao": 125, "precisao": "bf16", "semente": 42, "wd": 0.01}`
- AUROC por classe: NORM 0.9480, MI 0.9307, STTC 0.9262, CD 0.9352, HYP 0.9034
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/cnn_xresnet1d101_exploratory/metrics.json`

## 12. cnn-xresnet1d101

**Pergunta:** O baseline profundo de configuracao congelada reproduz o benchmark no protocolo oficial?

**Achado:** AUROC macro 0,9304 no fold 10 contra 0,928 publicado (Strodthoff et al. 2021): reproduz. Segundo uso do fold 10, sob a excecao para concorrentes de configuracao fixa (CLAUDE.md 6.1, 29/out). A diferenca exploratorio-confirmatorio e +0,002, como no catch22 (+0,003).

- Regime `confirmatory`: treino [1, 2, 3, 4, 5, 6, 7, 8], validação fold 9, teste fold 10
- Famílias: raw_signal
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"agregacao": "max", "arquitetura": "xresnet1d101", "early_stopping": null, "epocas": 50, "janela": 250, "lote": 128, "lr_max": 0.01, "passo_avaliacao": 125, "precisao": "bf16", "semente": 42, "wd": 0.01}`
- AUROC por classe: NORM 0.9475, MI 0.9369, STTC 0.9359, CD 0.9274, HYP 0.9043
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/cnn_xresnet1d101/metrics.json`

## 13. all-views-12leads-widegrid

**Pergunta:** Com logistica e boosting bem ajustados, o MLP ainda supera os modelos rasos nas features combinadas?

**Achado:** Boosting ajustado 0,9007 no fold 9, contra 0,8993 +- 0,0013 do MLP por semente (-0,001) e 0,9030 do MLP em ensemble de 5. Nao ha vantagem do MLP no conjunto completo. Ganho das visoes sobre o catch22: +0,024 sob boosting ajustado (0,8768 -> 0,9007) e +0,033 sob MLP por semente.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting (2 antigas + 10 sorteadas: lr, folhas, min_samples_leaf, l2, max_features), early stopping no fold 8", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9220, MI 0.9070, STTC 0.9172, CD 0.9041, HYP 0.8532
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/todas_as_visoes.json`

## 14. catch22-12leads-widegrid

**Pergunta:** Quanto o catch22 ganha com uma grade de boosting mais ampla e early stopping no fold 8?

**Achado:** 0,8768 contra 0,8715 da grade antiga (+0,005). As melhores configuracoes usam max_features=0,5 ou l2=1, ausentes da grade antiga. Escolheu lr=0,03 e 127 folhas, borda da grade. Supera o MLP por semente em 0,011.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting (2 antigas + 10 sorteadas: lr, folhas, min_samples_leaf, l2, max_features), early stopping no fold 8", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9061, MI 0.8844, STTC 0.9006, CD 0.8935, HYP 0.7994
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/catch22.json`

## 15. graph-view-12leads-widegrid

**Pergunta:** Com boosting ajustado, a visao de grafo ainda empata com o catch22?

**Achado:** 0,8794 contra 0,8733 da grade antiga (+0,006). Supera o catch22 ajustado (0,8768) em so +0,003. Sob MLP por semente o grafo supera o catch22 em +0,018 (0,8839 contra 0,8660): a vantagem do grafo depende do leitor. MLP sobre boosting ajustado: +0,004.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting (2 antigas + 10 sorteadas: lr, folhas, min_samples_leaf, l2, max_features), early stopping no fold 8", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9126, MI 0.8826, STTC 0.9116, CD 0.8832, HYP 0.8073
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/visao_grafo.json`

## 16. image-view-12leads-widegrid

**Pergunta:** Com logistica e boosting ajustados, a visao de imagem sozinha supera o catch22?

**Achado:** 0,8796 com boosting (a logistica deixou de vencer: na grade antiga ganhava com C=0,01 no piso). Supera o catch22 ajustado em +0,003. MLP por semente 0,8846 (+0,005).

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting (2 antigas + 10 sorteadas: lr, folhas, min_samples_leaf, l2, max_features), early stopping no fold 8", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9100, MI 0.8781, STTC 0.8907, CD 0.8751, HYP 0.8443
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/visao_imagem.json`

## 17. image-graph-12leads-widegrid

**Pergunta:** Sem o catch22, o MLP ainda supera o boosting ajustado?

**Achado:** Boosting ajustado 0,8925, MLP por semente 0,8966 (+0,004). Padrao em 17 conjuntos: o MLP supera o boosting ajustado em todos os conjuntos sem catch22 (+0,003 a +0,015) e empata ou perde em todos com catch22 (-0,001 a -0,011). Sem bootstrap pareado ainda.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting (2 antigas + 10 sorteadas: lr, folhas, min_samples_leaf, l2, max_features), early stopping no fold 8", "lphvg_rho": 1, "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9163, MI 0.8902, STTC 0.9084, CD 0.8940, HYP 0.8537
- Código: src e1767e681fed17ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/imagem_grafo.json`

## 18. series-amplitude-12leads

**Pergunta:** Todas as familias de imagem e grafo sao invariantes a escala; a voltagem sozinha (60 estatisticas em mV) carrega o sinal de HYP que falta?

**Achado:** 0,8568 macro, pior que o catch22 (0,8768) em tudo exceto HYP: 0,880 contra 0,799 (+0,081), a 0,023 da CNN exploratoria (0,903). Logistica so 0,809 de validacao contra 0,868 do boosting: a voltagem entra de forma nao-linear.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: series_amplitude
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.8854, MI 0.8157, STTC 0.8720, CD 0.8302, HYP 0.8804
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/series_amplitude.json`

## 19. series-interlead-12leads

**Pergunta:** Relacoes entre derivacoes (correlacoes, PCA, direcao do dipolo) carregam sinal que o tratamento univariado perde?

**Achado:** 0,7908 sozinha, fraca em todas as classes; MI 0,811 sem destaque. Nao sustenta que a perda seja do tratamento univariado.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: series_interlead
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.8374, MI 0.8113, STTC 0.7376, CD 0.8122, HYP 0.7554
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/series_interlead.json`

## 20. series-placebo-12leads

**Pergunta:** Controle: 99 colunas gaussianas sozinhas dao acaso?

**Achado:** 0,4964, acaso, como deveria.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: series_placebo
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas (C 1e-3..10) + 1 boosting com a config escolhida na grade ampla para o catch22 (lr 0,03, 127 folhas, msl 80, l2 0, mf 0,5), early stopping no fold 8; OMP_NUM_THREADS=4", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.4826, MI 0.4915, STTC 0.5028, CD 0.5044, HYP 0.5007
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_exploratory/series_placebo.json`

## 21. catch22-amplitude-12leads

**Pergunta:** A amplitude acrescenta ao catch22, que e invariante a escala?

**Achado:** 0,9026 contra 0,8768 do catch22 (+0,026) e 0,8741 do placebo de dimensao igual. HYP 0,799 -> 0,886 (+0,087); NORM, MI e STTC tambem sobem +0,012 a +0,015. Empata com todas as 10 visoes juntas (0,9007). A perda principal das visoes era a voltagem.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas (C 1e-3..10) + 1 boosting com a config escolhida na grade ampla para o catch22 (lr 0,03, 127 folhas, msl 80, l2 0, mf 0,5), early stopping no fold 8; OMP_NUM_THREADS=4", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9204, MI 0.8993, STTC 0.9131, CD 0.8937, HYP 0.8864
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_exploratory/catch22_amplitude.json`

## 22. catch22-interlead-12leads

**Pergunta:** As relacoes entre derivacoes acrescentam ao catch22?

**Achado:** 0,8844 (+0,008 sobre o catch22); MI +0,005. Ganho pequeno; o tratamento univariado pesa pouco.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_interlead
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas (C 1e-3..10) + 1 boosting com a config escolhida na grade ampla para o catch22 (lr 0,03, 127 folhas, msl 80, l2 0, mf 0,5), early stopping no fold 8; OMP_NUM_THREADS=4", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9119, MI 0.8890, STTC 0.9049, CD 0.9006, HYP 0.8154
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_exploratory/catch22_interderivacao.json`

## 23. catch22-amplitude-interlead-12leads

**Pergunta:** Base de serie forte: catch22 + amplitude + interderivacao, sem nenhuma visao de imagem ou grafo.

**Achado:** 0,9053, so +0,003 sobre catch22 + amplitude (dentro do ruido de um fold, sem bootstrap). Ja supera todas as visoes juntas (0,9007) com 363 features contra 3198.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas (C 1e-3..10) + 1 boosting com a config escolhida na grade ampla para o catch22 (lr 0,03, 127 folhas, msl 80, l2 0, mf 0,5), early stopping no fold 8; OMP_NUM_THREADS=4", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9251, MI 0.8985, STTC 0.9146, CD 0.8991, HYP 0.8892
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_exploratory/catch22_amplitude_interderivacao.json`

## 24. catch22-placebo-12leads

**Pergunta:** Controle da secao 6.3: 99 colunas aleatorias somadas ao catch22.

**Achado:** 0,8741, -0,003 sobre o catch22. O placebo nao acrescenta, entao os ganhos de amplitude e interderivacao nao sao efeito de dimensionalidade.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_placebo
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas (C 1e-3..10) + 1 boosting com a config escolhida na grade ampla para o catch22 (lr 0,03, 127 folhas, msl 80, l2 0, mf 0,5), early stopping no fold 8; OMP_NUM_THREADS=4", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9045, MI 0.8796, STTC 0.9003, CD 0.8913, HYP 0.7946
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_exploratory/catch22_placebo.json`

## 25. all-views-amplitude-interlead-12leads

**Pergunta:** Com a voltagem na base, as 10 visoes invariantes ainda acrescentam?

**Achado:** 0,9147, melhor numero nosso; a 0,014 da CNN exploratoria (0,9287), com CD a maior distancia (-0,023). Mas as 10 visoes acrescentam so +0,009 sobre catch22 + amp + inter (0,9053): o ganho das visoes sobre o catch22 (+0,024) encolhe quando a amplitude entra. Contra H1 na forma atual.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas + 1 boosting com a config escolhida na grade ampla para todas as visoes (lr 0,06, 31 folhas, msl 20, l2 1, mf 0,5), early stopping no fold 8", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9330, MI 0.9157, STTC 0.9197, CD 0.9123, HYP 0.8926
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_todas_exploratory/todas_amplitude_interderivacao.json`

## 26. all-views-placebo-12leads

**Pergunta:** Controle: placebo de mesma dimensao somado a todas as visoes.

**Achado:** 0,9022 contra 0,9007 sem placebo (+0,0015, ruido). Os +0,014 de amplitude + interderivacao sobre todas as visoes sao reais.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_placebo, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "grade": "fixa: 5 logisticas + 1 boosting com a config escolhida na grade ampla para todas as visoes (lr 0,06, 31 folhas, msl 20, l2 1, mf 0,5), early stopping no fold 8", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9225, MI 0.9076, STTC 0.9151, CD 0.9110, HYP 0.8549
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_fixed_todas_exploratory/todas_placebo.json`

## 27. catch22-amplitude-12leads-mlp

**Pergunta:** O MLP, que extraia mais das visoes que o boosting, muda a conclusao sobre amplitude e visoes?

**Achado:** Ensemble 0,8934 (por semente 0,8902) contra 0,8697 do catch22 no MLP (+0,024); HYP 0,876. Mesmo padrao do boosting, que fica acima (0,9026).

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "mlp": "6 configs escolhidas no fold 8 com 1 semente, depois 5 sementes; AdamW lr 3e-4, early stopping na AUROC de validacao (paciencia 20), corte +-5 DP", "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9164, MI 0.8867, STTC 0.9053, CD 0.8825, HYP 0.8762
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_amplitude.json`

## 28. catch22-interlead-12leads-mlp

**Pergunta:** O MLP, que extraia mais das visoes que o boosting, muda a conclusao sobre amplitude e visoes?

**Achado:** Ensemble 0,8768 (+0,007 sobre o catch22 no MLP). Ganho pequeno, como no boosting.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_interlead
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "mlp": "6 configs escolhidas no fold 8 com 1 semente, depois 5 sementes; AdamW lr 3e-4, early stopping na AUROC de validacao (paciencia 20), corte +-5 DP", "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9084, MI 0.8846, STTC 0.8997, CD 0.8800, HYP 0.8114
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_interderivacao.json`

## 29. catch22-amplitude-interlead-12leads-mlp

**Pergunta:** O MLP, que extraia mais das visoes que o boosting, muda a conclusao sobre amplitude e visoes?

**Achado:** Ensemble 0,8958 (por semente 0,8931); boosting 0,9053 no mesmo conjunto.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "mlp": "6 configs escolhidas no fold 8 com 1 semente, depois 5 sementes; AdamW lr 3e-4, early stopping na AUROC de validacao (paciencia 20), corte +-5 DP", "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9221, MI 0.8886, STTC 0.9075, CD 0.8861, HYP 0.8746
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_amplitude_interderivacao.json`

## 30. catch22-placebo-12leads-mlp

**Pergunta:** Controle: placebo somado ao catch22, no MLP.

**Achado:** Ensemble 0,8655, -0,004 sobre o catch22 no MLP. O placebo nao acrescenta.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_placebo
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "mlp": "6 configs escolhidas no fold 8 com 1 semente, depois 5 sementes; AdamW lr 3e-4, early stopping na AUROC de validacao (paciencia 20), corte +-5 DP", "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9002, MI 0.8699, STTC 0.8947, CD 0.8705, HYP 0.7924
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_placebo.json`

## 31. all-views-amplitude-interlead-12leads-mlp

**Pergunta:** O MLP, que extraia mais das visoes que o boosting, muda a conclusao sobre amplitude e visoes?

**Achado:** Ensemble 0,9089 (por semente 0,9059); boosting 0,9147. As visoes acrescentam +0,013 sobre catch22 + amp + inter no MLP (boosting +0,009); sobre o catch22 sozinho eram +0,033. Amplitude e visoes disputam parte do mesmo ganho de HYP. O leitor nao muda a conclusao.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "mlp": "6 configs escolhidas no fold 8 com 1 semente, depois 5 sementes; AdamW lr 3e-4, early stopping na AUROC de validacao (paciencia 20), corte +-5 DP", "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9331, MI 0.9062, STTC 0.9188, CD 0.9084, HYP 0.8776
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/todas_amplitude_interderivacao.json`

## 32. all-views-placebo-12leads-mlp

**Pergunta:** Controle: placebo somado a todas as visoes, no MLP.

**Achado:** Ensemble 0,9021 contra 0,9030 sem placebo (-0,001). O placebo nao acrescenta.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_placebo, image_spectrogram, image_scalogram, image_gasf, image_gadf, image_mtf, image_recurrence, graph_hvg, graph_lphvg1, graph_takens, graph_transition
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"amplitude": "por derivacao, em mV, sem normalizacao: media, desvio, pico a pico, p05, p95 (60)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "interlead": "8 derivacoes independentes: 28 correlacoes, fracao de variancia de 3 PCs, |carga| da PC1 (39)", "lphvg_rho": 1, "mlp": "6 configs escolhidas no fold 8 com 1 semente, depois 5 sementes; AdamW lr 3e-4, early stopping na AUROC de validacao (paciencia 20), corte +-5 DP", "nota_codigo": "rodado antes da adicao das familias _abs em src/; o comportamento padrao dos operadores nao mudou, mas src_sha256 abaixo e o do codigo posterior", "placebo": "99 gaussianas padrao, semente 42", "recurrence_rate": 0.05, "theiler_window": 1, "transition_n_bins": 10}`
- AUROC por classe: NORM 0.9296, MI 0.9022, STTC 0.9164, CD 0.9080, HYP 0.8542
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/todas_placebo.json`

## 33. image-spectrogram-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,8493 contra 0,8184 invariante (+0,031); HYP 0,775 -> 0,873. MLP: +0,029.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8867, MI 0.8204, STTC 0.8280, CD 0.8381, HYP 0.8731
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/image_spectrogram_abs.json`

## 34. image-scalogram-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,8650 contra 0,8480 (+0,017); HYP 0,796 -> 0,876. MLP: +0,018.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_scalogram_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9035, MI 0.8359, STTC 0.8679, CD 0.8425, HYP 0.8755
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/image_scalogram_abs.json`

## 35. image-gasf-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,8700 contra 0,8287 (+0,041), o maior ganho; HYP 0,753 -> 0,859. MLP: +0,045.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_gasf_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8967, MI 0.8603, STTC 0.8783, CD 0.8559, HYP 0.8588
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/image_gasf_abs.json`

## 36. image-gadf-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,8573 contra 0,8498 (+0,007); HYP 0,785 -> 0,812. Ganho pequeno; MLP +0,005.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_gadf_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8975, MI 0.8538, STTC 0.8750, CD 0.8483, HYP 0.8117
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/image_gadf_abs.json`

## 37. image-mtf-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,7657 contra 0,7289 (+0,037); HYP 0,676 -> 0,740. Continua a familia mais fraca. MLP: +0,043.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_mtf_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8150, MI 0.7655, STTC 0.7722, CD 0.7353, HYP 0.7405
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/image_mtf_abs.json`

## 38. image-recurrence-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,8184 contra 0,8180 (0,000); HYP sobe 0,711 -> 0,759 mas o resto cai. Com epsilon fixo, 2-10% dos registros ficam com taxa < 0,001, justamente os de amplitude alta. MLP: -0,005.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_recurrence_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8705, MI 0.7967, STTC 0.8505, CD 0.8148, HYP 0.7594
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/image_recurrence_abs.json`

## 39. graph-takens-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,8112 contra 0,8371 (-0,026): piora. As redes de registros de amplitude alta ficam quase vazias com epsilon fixo; 4.887 NaN. MLP: -0,028. Epsilon fixo nao serve para a leitura por rede.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_takens_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8640, MI 0.7851, STTC 0.8320, CD 0.8006, HYP 0.7741
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/graph_takens_abs.json`

## 40. graph-transition-abs-12leads

**Pergunta:** Trocar a normalizacao por amostra por constantes de treino devolve a amplitude a esta familia e melhora a AUROC?

**Achado:** 0,7976 contra 0,7633 (+0,034); HYP 0,678 -> 0,758. MLP: +0,023.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.8416, MI 0.8096, STTC 0.8192, CD 0.7596, HYP 0.7581
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/graph_transition_abs.json`

## 41. image-view-abs-12leads

**Pergunta:** A visao de imagem com amplitude supera a invariante?

**Achado:** 0,8924 contra 0,8796 (+0,013); HYP 0,844 -> 0,888. MLP 0,9014, que empata com catch22 + todas as visoes invariantes (0,9030) sem catch22.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, image_recurrence_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9188, MI 0.8726, STTC 0.9049, CD 0.8775, HYP 0.8881
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/visao_imagem_abs.json`

## 42. graph-view-abs-12leads

**Pergunta:** A visao de grafo com transicao e Takens _abs (HVG e LPHVG ordinais, inalteradas) supera a invariante?

**Achado:** 0,8780 contra 0,8794 (-0,001): o ganho da transicao e anulado pela perda da Takens. MLP: -0,002.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_hvg, graph_lphvg1, graph_takens_abs, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9128, MI 0.8806, STTC 0.9012, CD 0.8693, HYP 0.8261
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/visao_grafo_abs.json`

## 43. catch22-views-abs-12leads

**Pergunta:** Sem a familia de amplitude, as visoes _abs carregam a voltagem sozinhas?

**Achado:** 0,9106 contra 0,9007 das visoes invariantes (+0,010); HYP 0,853 -> 0,889. Quase o mesmo que catch22 + amp + inter + visoes invariantes (0,9147) sem nenhuma feature de amplitude explicita. MLP 0,9089 contra 0,9030.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, image_recurrence_abs, graph_hvg, graph_lphvg1, graph_takens_abs, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9314, MI 0.9115, STTC 0.9153, CD 0.9057, HYP 0.8893
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/catch22_visoes_abs.json`

## 44. catch22-amplitude-interlead-views-abs-12leads

**Pergunta:** Com a amplitude na base, as visoes _abs acrescentam mais que as invariantes?

**Achado:** 0,9113 contra 0,9147 com visoes invariantes (-0,003, ruido). A amplitude pela familia explicita e pelas visoes _abs e a mesma informacao: juntas nao somam. Sobre catch22 + amp + inter (0,9046 na grade ampla, 0,9053 na fixa) as visoes acrescentam ~0,01 em qualquer versao. Nao sustenta H1 na forma de ganho de AUROC.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, image_recurrence_abs, graph_hvg, graph_lphvg1, graph_takens_abs, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9305, MI 0.9109, STTC 0.9157, CD 0.9038, HYP 0.8957
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/catch22_amplitude_interderivacao_visoes_abs.json`

## 45. catch22-amplitude-interlead-views-abs-norec-12leads

**Pergunta:** Tirar as duas leituras de recorrencia _abs, que degeneram, ajuda?

**Achado:** 0,9128 contra 0,9113 com elas (+0,0015, ruido de um fold); HYP 0,896. Ainda abaixo de 0,9147 com visoes invariantes.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, graph_hvg, graph_lphvg1, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "ampla: C 1e-3..10; 12 boosting, early stopping no fold 8; OMP_NUM_THREADS=4", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9340, MI 0.9103, STTC 0.9165, CD 0.9068, HYP 0.8963
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_wide_exploratory/catch22_amplitude_interderivacao_visoes_abs_sem_recorrencia.json`

## 46. image-view-abs-12leads-mlp

**Pergunta:** Sob o MLP, as visoes _abs mudam a conclusao do boosting?

**Achado:** Ensemble 0,9014 (por semente 0,8981) contra 0,8883 invariante (+0,013); HYP 0,839 -> 0,882.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, image_recurrence_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "MLP: 6 configs no fold 8 com 1 semente, depois ensemble de 5 sementes", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9314, MI 0.8890, STTC 0.9106, CD 0.8937, HYP 0.8825
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/visao_imagem_abs.json`

## 47. graph-view-abs-12leads-mlp

**Pergunta:** Sob o MLP, as visoes _abs mudam a conclusao do boosting?

**Achado:** Ensemble 0,8850 contra 0,8874 invariante (-0,002), como no boosting.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_hvg, graph_lphvg1, graph_takens_abs, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "MLP: 6 configs no fold 8 com 1 semente, depois ensemble de 5 sementes", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9220, MI 0.8833, STTC 0.9124, CD 0.8800, HYP 0.8274
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/visao_grafo_abs.json`

## 48. catch22-views-abs-12leads-mlp

**Pergunta:** Sob o MLP, as visoes _abs mudam a conclusao do boosting?

**Achado:** Ensemble 0,9089 contra 0,9030 com visoes invariantes (+0,006); igual a catch22 + amp + inter + visoes invariantes no MLP (0,9089).

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, image_recurrence_abs, graph_hvg, graph_lphvg1, graph_takens_abs, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "MLP: 6 configs no fold 8 com 1 semente, depois ensemble de 5 sementes", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9338, MI 0.9051, STTC 0.9193, CD 0.9022, HYP 0.8841
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_visoes_abs.json`

## 49. catch22-amplitude-interlead-views-abs-12leads-mlp

**Pergunta:** Sob o MLP, as visoes _abs mudam a conclusao do boosting?

**Achado:** Ensemble 0,9109 contra 0,9089 com visoes invariantes (+0,002). A familia de amplitude acrescenta so +0,002 sobre catch22 + visoes _abs.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, image_recurrence_abs, graph_hvg, graph_lphvg1, graph_takens_abs, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "MLP: 6 configs no fold 8 com 1 semente, depois ensemble de 5 sementes", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9358, MI 0.9079, STTC 0.9206, CD 0.9034, HYP 0.8866
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_amplitude_interderivacao_visoes_abs.json`

## 50. catch22-amplitude-interlead-views-abs-norec-12leads-mlp

**Pergunta:** Sob o MLP, as visoes _abs mudam a conclusao do boosting?

**Achado:** Ensemble 0,9123 (por semente 0,9095), melhor numero do MLP; +0,017 sobre catch22 + amp + inter (0,8958) contra +0,013 das visoes invariantes. Diferenca pequena e de um fold so.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, image_spectrogram_abs, image_scalogram_abs, image_gasf_abs, image_gadf_abs, image_mtf_abs, graph_hvg, graph_lphvg1, graph_transition_abs
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"calibracao": "data/features/absolute_calibration.json, so folds 1-7 (scripts/calibrate_absolute_views.py)", "embedding": "data/features/takens_embedding.json (m 7-12, tau 9-12)", "epsilon": "percentil 10 entre registros do limiar de taxa 0,05; o mesmo para RQA e Takens (matrizes identicas)", "grade": "MLP: 6 configs no fold 8 com 1 semente, depois ensemble de 5 sementes", "haralick": "16 niveis, 15 cortes fixos dos pixels de imagens de treino", "lphvg_rho": 1, "mtf_edges": "8 estados, quantis do conjunto de amostras de treino", "scale": "p99,9 do |x| maximo por registro, por derivacao; GAF corta em +-1", "theiler_window": 1, "transition_edges": "10 estados, idem"}`
- AUROC por classe: NORM 0.9369, MI 0.9091, STTC 0.9216, CD 0.9030, HYP 0.8910
- Código: src 47527679d43b123c
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_mlp_exploratory/catch22_amplitude_interderivacao_visoes_abs_sem_recorrencia.json`

## 51. multiplex-hvg-lphvg-8leads

**Pergunta:** Uma leitura ENTRE derivacoes (VG multiplex de Lacasa et al. 2015, 8 camadas) separa as superclasses sozinha?

**Achado:** AUROC macro 0,8465 no fold 9 com 196 features ordinais, abaixo do catch22 (0,8768). HVG sozinha 0,8237 e LPHVG sozinha 0,8293; juntas somam +0,017 sobre a LPHVG, mais do que o null de reparametrizacao preve (+0,001 nas familias univariadas), sem bootstrap.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: graph_hvg_multiplex, graph_lphvg1_multiplex
- Derivações: I, II, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"boosting": {"l2_regularization": 0.0, "learning_rate": 0.03, "max_features": 0.5, "max_leaf_nodes": 127, "min_samples_leaf": 80}, "jaccard_excludes_consecutive_edges": true, "layers": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"], "lphvg_rho": 1, "mi_degree_cap": 16}`
- AUROC por classe: NORM 0.8935, MI 0.8561, STTC 0.8638, CD 0.8444, HYP 0.7747
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_key_coalitions_exploratory/multiplex.json`

## 52. catch22-amplitude-multiplex

**Pergunta:** O multiplex substitui a interderivacao (28 correlacoes de Pearson + PCA) como leitura das relacoes entre derivacoes?

**Achado:** 0,9085 no fold 9 contra 0,9053 com a interderivacao no lugar: +0,0031, IC 95% [+0,0003; +0,0061] (bootstrap pareado, 1000). Substitui com ganho pequeno, IC rente a zero.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, graph_hvg_multiplex, graph_lphvg1_multiplex
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"boosting": {"l2_regularization": 0.0, "learning_rate": 0.03, "max_features": 0.5, "max_leaf_nodes": 127, "min_samples_leaf": 80}, "jaccard_excludes_consecutive_edges": true, "layers": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"], "lphvg_rho": 1, "mi_degree_cap": 16}`
- AUROC por classe: NORM 0.9276, MI 0.9038, STTC 0.9172, CD 0.9008, HYP 0.8930
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_key_coalitions_exploratory/catch22_amplitude_multiplex.json`

## 53. base-multiplex

**Pergunta:** O multiplex acrescenta a base de serie (catch22 + amplitude + interderivacao)?

**Achado:** 0,9097 no fold 9: +0,0044, IC 95% [+0,0018; +0,0071] sobre a base. A interderivacao nao soma nada com o multiplex presente (+0,0012, IC [-0,0007; +0,0030]). O direcionado soma +0,012 a mesma base, quase 3x mais.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, graph_hvg_multiplex, graph_lphvg1_multiplex
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"boosting": {"l2_regularization": 0.0, "learning_rate": 0.03, "max_features": 0.5, "max_leaf_nodes": 127, "min_samples_leaf": 80}, "jaccard_excludes_consecutive_edges": true, "layers": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"], "lphvg_rho": 1, "mi_degree_cap": 16}`
- AUROC por classe: NORM 0.9279, MI 0.9008, STTC 0.9221, CD 0.9029, HYP 0.8948
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_key_coalitions_exploratory/base_multiplex.json`

## 54. base-directed-multiplex

**Pergunta:** O multiplex acrescenta sobre base + direcionado (o candidato podado)?

**Achado:** 0,9170 no fold 9 contra 0,9169: +0,0000, IC 95% [-0,0017; +0,0018]. Com o direcionado presente o ganho do multiplex some; a leitura univariada em mV nas 12 derivacoes ja cobre o que ele acrescentava a base. Mais um caso de substituibilidade.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, graph_hvg_dir, graph_lphvg1_dir, graph_hvg_dir_mv, graph_lphvg1_dir_mv, graph_hvg_multiplex, graph_lphvg1_multiplex
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"boosting": {"l2_regularization": 0.0, "learning_rate": 0.03, "max_features": 0.5, "max_leaf_nodes": 127, "min_samples_leaf": 80}, "jaccard_excludes_consecutive_edges": true, "layers": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"], "lphvg_rho": 1, "mi_degree_cap": 16}`
- AUROC por classe: NORM 0.9371, MI 0.9188, STTC 0.9211, CD 0.9122, HYP 0.8954
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_key_coalitions_exploratory/base_direcionado_multiplex.json`

## 55. best-set-multiplex

**Pergunta:** O multiplex acrescenta ao melhor conjunto (base + visibilidade + line graph + mv + direcionado, 0,9199)?

**Achado:** 0,9210 no fold 9, o maior teste ate aqui, mas +0,0012, IC 95% [-0,0004; +0,0029]: nao significativo, abaixo do MDE (~0,003). Unico sinal por classe e MI (+0,004, 0,9187 -> 0,9225), coerente com localizacao de territorio, sem IC por classe.

- Regime `exploratory`: treino [1, 2, 3, 4, 5, 6, 7], validação fold 8, teste fold 9
- Famílias: catch22, series_amplitude, series_interlead, graph_hvg, graph_lphvg1, graph_hvg_line, graph_lphvg1_line, graph_hvg_line_mv, graph_lphvg1_line_mv, graph_hvg_dir, graph_lphvg1_dir, graph_hvg_dir_mv, graph_lphvg1_dir_mv, graph_hvg_multiplex, graph_lphvg1_multiplex
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"boosting": {"l2_regularization": 0.0, "learning_rate": 0.03, "max_features": 0.5, "max_leaf_nodes": 127, "min_samples_leaf": 80}, "jaccard_excludes_consecutive_edges": true, "layers": ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"], "lphvg_rho": 1, "mi_degree_cap": 16}`
- AUROC por classe: NORM 0.9425, MI 0.9225, STTC 0.9263, CD 0.9152, HYP 0.8987
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/views_key_coalitions_exploratory/base_visibilidade_line_graph_mv_direcionado_multiplex.json`

## 56. lstm-bidir-confirmatory

**Pergunta:** Baseline de configuracao fixa (Strodthoff et al. 2021, sem early stopping) no fold 10 selado, sob a excecao de 29/out.

**Achado:** 5 sementes: 0,9205 +- 0,0004 no fold 10 (por semente 0,9200 / 0,9210 / 0,9204 / 0,9207 / 0,9206), dentro do publicado (0,921 +- 0,006). Porte sem fastai reproduz o benchmark.

- Regime `confirmatory`: treino [1, 2, 3, 4, 5, 6, 7, 8], validação fold 9, teste fold 10
- Famílias: raw_signal
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"arch": "lstm_bidir", "crop": 250, "epochs": 50, "hidden": 256, "layers": 2, "max_lr": 0.001, "seeds": [42, 43, 44, 45, 46]}`
- AUROC por classe: NORM 0.9425, MI 0.9216, STTC 0.9262, CD 0.9164, HYP 0.8933
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/deep_lstm_bidir_confirmatory_seed42/metrics.json`

## 57. inception1d-confirmatory

**Pergunta:** Baseline de configuracao fixa (Strodthoff et al. 2021, sem early stopping) no fold 10 selado, sob a excecao de 29/out.

**Achado:** 5 sementes: 0,9188 +- 0,0015 no fold 10 (0,9197 / 0,9191 / 0,9173 / 0,9207 / 0,9174), dentro do publicado (0,921 +- 0,013).

- Regime `confirmatory`: treino [1, 2, 3, 4, 5, 6, 7, 8], validação fold 9, teste fold 10
- Famílias: raw_signal
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"arch": "inception1d", "crop": 250, "depth": 6, "epochs": 50, "kernel_size": 40, "max_lr": 0.01, "seeds": [42, 43, 44, 45, 46]}`
- AUROC por classe: NORM 0.9411, MI 0.9171, STTC 0.9254, CD 0.9171, HYP 0.8975
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/deep_inception1d_confirmatory_seed42/metrics.json`

## 58. transformer-deepecgkit-confirmatory

**Pergunta:** Transformer do DeepECG-Kit no fold 10, decidido pelo Eduardo em 09/nov com ressalva declarada.

**Achado:** 5 sementes: 0,9004 +- 0,0028 no fold 10 (0,9025 / 0,9028 / 0,8960 / 0,8997 / 0,9008). RESSALVA: configuracao = padroes do repositorio, nao uma configuracao publicada para PTB-XL; treinado com a receita do benchmark (one-cycle, crops de 2,5 s).

- Regime `confirmatory`: treino [1, 2, 3, 4, 5, 6, 7, 8], validação fold 9, teste fold 10
- Famílias: raw_signal
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"arch": "transformer", "crop": 250, "d_model": 128, "epochs": 50, "ff": 512, "heads": 8, "layers": 4, "max_lr": 0.001, "seeds": [42, 43, 44, 45, 46], "source": "DeepECG-Kit@5ecd8c17"}`
- AUROC por classe: NORM 0.9300, MI 0.8966, STTC 0.9112, CD 0.8834, HYP 0.8910
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/deep_transformer_confirmatory_seed42/metrics.json`

## 59. xresnet1d101-confirmatory-5seeds

**Pergunta:** Variancia entre sementes da xresnet1d101 no fold 10 (a semente 42 ja esta registrada como cnn-xresnet1d101).

**Achado:** 5 sementes: 0,9283 +- 0,0013 no fold 10 (0,9304 / 0,9286 / 0,9276 / 0,9272 / 0,9276); a semente 42 registrada antes e a mais alta das cinco. Dentro do publicado (0,925 +- 0,007).

- Regime `confirmatory`: treino [1, 2, 3, 4, 5, 6, 7, 8], validação fold 9, teste fold 10
- Famílias: raw_signal
- Derivações: I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6
- Parâmetros: `{"arch": "xresnet1d101", "crop": 250, "epochs": 50, "max_lr": 0.01, "seeds": [42, 43, 44, 45, 46]}`
- AUROC por classe: NORM 0.9489, MI 0.9344, STTC 0.9357, CD 0.9233, HYP 0.9006
- Código: src be1c0f4992bad7ac
- Métricas completas: `/exp/duducampos/InterpretECG/results/cnn_xresnet1d101_seed43/metrics.json`
