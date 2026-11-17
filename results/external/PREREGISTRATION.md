# Pré-registro: generalização em datasets externos

Gravado em 10/nov/2026, antes de qualquer feature ou resultado dos datasets
abaixo. Decisão do Eduardo, na mesma data, após feedback do orientador: dois
datasets externos, CODE-15% e Chapman-Shaoxing (o INCART/São Petersburgo foi
descartado: 32 pacientes, anotação por batimento).

## Pergunta

As listas de features escolhidas no PTB-XL (exploratório) transferem para
outro hospital, outro equipamento e outra tarefa? Em particular, o conjunto
multi-view continua à frente do só-série com o mesmo número de features?

## Dados e divisão (scripts/prepare_external.py)

| dataset | registros | rótulos | divisão |
|---|---|---|---|
| Chapman-Shaoxing (Zheng et al. 2020) | 10.646, 1 por paciente | 4 grupos de ritmo: SB, AFIB, GSVT, SR | 10 folds estratificados, semente 42 |
| CODE-15% (Ribeiro et al.), partes 0–2 | exames de 10 s, 1 por paciente (sorteado, semente 42) | 1dAVb, RBBB, LBBB, SB, ST, AF | 10 folds estratificados por paciente, semente 42 |

Sinais reamostrados para 100 Hz (`resample_poly`), em mV, 12 derivações,
1.000 amostras. Treino folds 1–8, validação (early stopping) fold 9, teste
fold 10. O fold 10 de cada dataset é lido uma vez por modelo abaixo.

## O que é congelado

- Listas de features: `results/confirmatory/frozen_feature_lists.json`
  (tsfresh k=268, SHA 3c7c0d8d63fbf093; melhor + multiplex k=537,
  SHA 74b3396bd53d03dc). Nenhuma repoda, nenhuma feature nova.
- Extração: as mesmas funções do PTB-XL (`scripts/extract_external_features.py`,
  conferido contra os parquets do PTB-XL em 10 registros: diferença máxima 3e-7).
- Modelo: `BoostingPorRotulo`, configuração `BOOSTING` de
  `scripts/evaluate_augmented.py`, max_iter 2000, early stopping no fold 9,
  semente 42. Um classificador binário por rótulo (um contra o resto no Chapman).

## Análises

- AUROC macro (e por rótulo) no fold 10 para os dois conjuntos, nos dois datasets.
- Bootstrap pareado (1.000 reamostragens de registros do fold 10) de
  melhor + multiplex − tsfresh, por dataset.
- Todos os números são reportados, inclusive se o multi-view perder.
- Diferenças por rótulo são descritivas; nenhuma afirmação da H2 se apoia
  só nelas sem novo registro aqui.

## Registro de incidentes

- 10/nov/2026: a extração do Chapman e o download da parte 2 do CODE-15%
  caíram junto com a sessão; foram retomados (`results/logs/run_external.sh`).
  Nenhum dado de teste foi lido nesse intervalo.
- 11/nov/2026, **desvio de protocolo, decidido pelo Eduardo depois de ver o
  resultado das listas congeladas no Chapman** (multi-view − tsfresh =
  −0,0022 [−0,0061; +0,0014]): o PRISM é dependente da tarefa — gera todas as
  visões e poda na validação da própria tarefa. Aplicar a lista do PTB-XL
  testa a transferência da lista, não o método. Correção: em cada dataset,
  repetir a poda do PTB-XL (mesmo conjunto de partida — as 15 famílias e o
  tsfresh completo —, mesmo boosting, mesma grade de k, permutação com 2
  repetições, semente 42) com treino nos folds 1–7 e ranking/early stopping no
  fold 8; escolher k pelo maior AUROC no fold 8; depois reajustar com treino
  1–8, early stopping no 9 e testar no 10. O fold 10 não é lido durante a
  poda. Os dois resultados (lista transferida e poda por tarefa) são
  reportados; a mudança e sua motivação são declaradas no artigo.
- 11/nov/2026, **segunda tarefa no Chapman, decidida pelo Eduardo** antes de
  qualquer resultado da poda por tarefa: os 4 grupos ficam no teto (AUROC >
  0,98 nos dois conjuntos, como a literatura já reportava), o que tira do
  dataset o poder de separar representações independentemente de quem
  vença. Acrescenta-se `chapman8`: os 11 ritmos originais da coluna `Rhythm`,
  excluídas as classes com menos de 30 registros no total (AVNRT 16, AVRT 8,
  SAAWR 7; os 31 registros saem). Ficam SB, SR, AFIB, ST, SVT, AF, SA, AT, um
  contra o resto, mesmos folds do Chapman de 4 grupos, mesmo protocolo de poda
  e de avaliação. Os resultados dos 4 grupos continuam reportados.
- 11/nov/2026, **CODE-15% passa a ser o dataset inteiro, decidido pelo
  Eduardo antes de qualquer leitura do fold 10 do CODE-15%**. Motivo: poder
  estatístico. O subconjunto (partes 0–2, 21.174 registros) deixava 35–47
  positivos por classe no teste; o Chapman de 8 ritmos mostrou o que isso
  produz (a classe com 11 casos decidiu a média). Novo recorte: as 18 partes,
  mesmos filtros (exames de 10 s, 1 por paciente sorteado com semente 42),
  novos 10 folds estratificados por paciente. A poda multi-view do
  subconjunto (k=67 escolhido no fold 8) e a poda tsfresh interrompida não
  são usadas; nenhum modelo do subconjunto leu o fold 9 para avaliação nem o
  fold 10.
- 11/nov/2026, **poda a partir de todas as famílias, decidida pelo Eduardo**
  depois de ver os resultados do Chapman com a poda por tarefa (tsfresh à
  frente). O conjunto de partida "melhor + multiplex" (15 famílias) já vinha
  de um filtro de famílias feito no PTB-XL (Shapley e coalizões no
  exploratório), que descartou espectrograma, escalograma, GASF, GADF, MTF,
  RQA, rede de Takens e grafo de transição. Pela mesma lógica da poda por
  tarefa, esse filtro também é dependente da tarefa e não deve ser herdado.
  Acrescenta-se o conjunto `todas_familias` (as 15 + as 8 descartadas), podado
  em cada dataset pelo mesmo protocolo; "melhor + multiplex" e tsfresh
  continuam reportados. Embedding de Takens: o congelado do PTB-XL (folds
  1–7). As variantes `_abs` ficam de fora: dependem de constantes calibradas
  no treino do PTB-XL. Conferência da extração: as 6 famílias de textura e o
  grafo de transição batem exatamente com o PTB-XL; RQA e Takens diferem em
  ~0,01% das arestas porque os sinais externos estão em float32 e o limiar de
  5% de recorrência cai em distâncias quase empatadas (mesmo sinal em float64
  reproduz o valor do PTB-XL).
- 12/nov/2026: **rótulos do fold 10 compartilhados**, por decisão do Eduardo,
  num pacote dos três datasets (`export/ecg_fm_bundle/`, gerado por
  `scripts/export_fm_bundle.py`) entregue a outra pessoa para avaliar foundation
  models. O README do pacote fixa o protocolo (treino 1–8, validação 9, teste
  10 lido uma vez por configuração final) e pede as probabilidades por registro
  dos folds 9 e 10. As avaliações feitas por terceiros não passam pelo
  `results/test_fold_usage.jsonl`; o número de leituras do fold 10 informado
  por quem rodar será declarado no artigo junto com os resultados.
- 13/nov/2026: **CODE-15%, fold 10 lido uma vez** para as listas podadas
  tsfresh (k=537) e melhor + multiplex (k=134): 0,9676 contra 0,9746;
  diferença +0,0070 [+0,0021; +0,0122], p(≤0)=0,003 (`bootstrap_pruned.json`).
  Incidente operacional: um erro de monitoramento relançou a mesma avaliação
  duas vezes; as duas cópias foram mortas durante o ajuste, antes de prever o
  fold 10, e só a primeira execução produziu escores. O conjunto
  `todas_familias` do CODE-15% ainda depende da extração das 8 famílias extras.

## Acréscimo de 13/nov/2026: tabela final (decisão do Eduardo, antes de qualquer execução abaixo)

Para completar a tabela catch22 / tsfresh / nosso boosting / baselines
profundos / nossa melhor view profunda nos três datasets externos:

- **catch22 + boosting** (`scripts/eval_catch22_fold10.py`): as 264 features,
  sem seleção, boosting de configuração fixa, treino 1–8, early stopping no 9,
  fold 10 lido uma vez por dataset (também no PTB-XL).
- **Baselines profundos**, configuração congelada idêntica à do PTB-XL, sem
  nenhum ajuste para os novos dados; 5 sementes (42–46); treino 1–8,
  validação 9, teste 10:
  - xresnet1d101, inception1d, LSTM bidirecional, transformer
    (`scripts/train_cnn.py --dataset`): 50 épocas, sem early stopping;
  - ECG-IMN (`scripts/train_imn.py --dataset`): early stopping no fold 9,
    como no PTB-XL.
- **GNN em L(HVG) direcionada** (`scripts/train_line_gnn.py --dataset`), a
  mesma configuração do confirmatório do PTB-XL (hidden 64, 4 camadas,
  50 épocas, época escolhida no fold 9), 5 sementes. Cache da HVG por dataset
  (`scripts/build_hvg_cache.py --dataset`); o chapman8 usa o cache do chapman.
- Tarefas: chapman (4 grupos), chapman8 (8 ritmos), code15. Em todas, BCE
  um-contra-o-resto, como o boosting.
- Cada leitura do fold 10 vai para `results/test_fold_usage.jsonl` com o
  campo `dataset`. Todos os números são reportados.
- Mudanças de código, sem efeito no PTB-XL: opção `--dataset` (padrão
  `ptbxl`) e módulo `src/interpretecg/external.py`; na GNN, as arestas ficam
  em int16 na RAM e viram int64 na GPU (mesmos valores).
- 13/nov/2026, mesmo dia: **Chapman de 4 grupos sai da tabela final**, por
  decisão do Eduardo (tarefa no teto, todos os métodos empatados). A primeira
  rodada profunda nele (xresnet1d101, semente 42) foi interrompida na época
  ~20, antes de ler o fold 10, e seu diretório removido. O catch22 + boosting
  já tinha lido o fold 10 do Chapman de 4 grupos (0,9642) e continua registrado.
  Os baselines profundos e a GNN rodam só no chapman8 e no code15.
- 13/nov/2026: **sementes dos boosting**, decisão do Eduardo: catch22, tsfresh
  e a lista "nossa" reajustados com as sementes 42–46 (só a semente do
  HistGradientBoosting muda; listas e configuração fixas), no PTB-XL, chapman8
  e code15, para reportar média ± dp como nos modelos profundos
  (`scripts/boosting_seeds_fold10.py`). A semente 42 é conferida contra o
  número já gravado. Cada leitura do fold 10 vai para o `test_fold_usage.jsonl`.
- 13/nov/2026, **modelo profundo na melhor visão de cada dataset**, decisão do
  Eduardo: a linha "nossa visão + deep" usa a melhor família isolada **de cada
  tarefa**, não a do PTB-XL. Critério igual ao que escolheu L(HVG) mV no
  PTB-XL: cada família sozinha, boosting fixo, treino 1–7, early stopping e
  escolha no fold 8 (`scripts/external_single_views.py`,
  `results/external/<ds>/single_views.csv`); folds 9 e 10 não participam.
  - chapman8: vence `image_recurrence` (RQA, val 0,9406; L(HVG) mV fica em
    12º, 0,9084). Modelo: a mesma `HVGGNN` direcionada (2 relações, mesmos
    hiperparâmetros do PTB-XL) sobre a **rede de recorrência de Takens**, que
    é a mesma matriz do RP (embedding congelado, taxa 0,05, Theiler 1; cache
    conferido contra `graph_takens`: grau médio idêntico). 5 sementes,
    época escolhida no fold 9, fold 10 lido uma vez por semente.
  - code15: escolha adiada até a extração das 8 famílias extras (entre elas
    o RP); a avaliação das 15 já extraídas está registrada no mesmo CSV.
  - A GNN em L(HVG) direcionada continua rodando nos dois datasets, para a
    discussão ("a melhor visão muda com a tarefa?").
- 13/nov/2026, **correção de código antes de qualquer GNN externa**: a GNN
  era construída com 5 saídas fixas (as superclasses do PTB-XL); agora usa
  o número de rótulos do dataset. Nenhuma GNN externa tinha rodado.
- 14/nov/2026: **LSTM bidirecional no code15, semente 42, divergiu** (perda
  NaN entre as épocas 20 e 30; validação 0,63 na época 10 e 0,77 na 20). O
  script caiu na checagem de validação, antes de prever o fold 10, que não foi
  lido. Pela configuração congelada (sem early stopping, sem reinício), a
  rodada é reportada como falha; nenhuma semente substituta é sorteada. As
  outras 4 sementes: 0,9799; 0,9466; 0,9285; 0,9746.
- 14/nov/2026, decisão do Eduardo: a GNN na rede de Takens do chapman8 ficou
  abaixo do boosting (~0,950 contra 0,9685, 4 de 5 sementes). Nenhuma
  investigação até a fila inteira terminar. Depois, uma análise **post-hoc**,
  declarada como tal no artigo, procura a causa (candidatas, ainda não
  testadas: densidade da rede de recorrência e diluição por média na passagem
  de mensagens; registros degenerados com até ~420 mil arestas numa derivação;
  diferença entre a leitura RQA, por diagonais, e a leitura por vizinhança).
  Nenhum resultado post-hoc substitui o número pré-registrado na tabela.
- 15/nov/2026, **CODE-15%: `todas_familias` podado e lista "nossa"**, antes
  de qualquer leitura do fold 10 com esse conjunto. A extração das 8 famílias
  extras e a poda terminaram (margarita, 15/nov 07:17). Validação no fold 8
  (treino 1–7) das três listas podadas: tsfresh k=537, 0,9774; melhor +
  multiplex k=134, 0,9811; todas_familias k=268, 0,9840. Pela mesma regra que
  fez de `todas_familias` a lista "nossa" do chapman8 (maior AUROC no fold 8
  entre as listas podadas), ela passa a ser a lista "nossa" do CODE-15%.
  Leituras do fold 10: (1) `evaluate_external.py code15 --lists pruned --sets
  todas_familias`, uma vez; (2) sementes 42–46 da lista nova
  (`boosting_seeds_fold10.py`). Os números de melhor + multiplex no CODE-15%
  (0,9746; sementes 0,9748 ± 0,0004) continuam registrados e reportados.
- 15/nov/2026, **CODE-15%: melhor visão para o modelo profundo**, decisão do
  Eduardo antes de qualquer GNN ler o fold 10 do CODE-15%. Avaliação isolada
  das 23 famílias (`single_views.csv`, fold 8, treino 1–7): a primeira é o
  catch22 (0,9587), que é a própria série; o modelo profundo da série são os
  baselines no sinal cru, já na tabela. Regra adotada: a linha "melhor visão +
  deep" usa a **melhor visão derivada**, que no CODE-15% é o L(HVG) ordinal
  (`graph_hvg_line`, 0,9524), empatado com o L(LPHVG) ordinal
  (`graph_lphvg1_line`, 0,9523; diferença declarada no artigo). Modelo: a GNN
  em L(HVG) direcionada já na fila (mesma configuração do PTB-XL, 5 sementes),
  que passa a ser a linha pré-registrada do CODE-15%. No chapman8 e no PTB-XL
  a regra não muda nada (RP e L(HVG) mV já eram visões derivadas).
- 15/nov/2026, **análise post-hoc da GNN na rede de Takens (chapman8)**,
  antecipada por decisão do Eduardo (a margarita estava livre). Declarada
  como post-hoc; nada aqui substitui o 0,9501 ± 0,0035 da tabela. Só folds
  1–8 (treino 1–7, validação 8); o fold 10 não é lido. Scripts
  `scripts/posthoc_takens_{edge_stats,spans,smoothing,io}.py`; saídas em
  `results/external/chapman8/posthoc_takens/`.
  - Arestas (240 registros de treino × 12 derivações): Takens tem mediana de
    21.300 arestas por derivação (grau ~43) contra 1.979 da HVG; só 4,5% têm
    vão ≤ 5 amostras e 20% ficam dentro da janela de embedding, ou seja, a
    maioria liga batimentos diferentes. Nenhuma derivação com > 100 mil
    arestas na amostra: a hipótese dos registros degenerados não explica uma
    diferença sistemática.
  - Informação nos vãos (boosting fixo, validação fold 8): histograma
    completo dos vãos da Takens (23 faixas × 12) 0,9418, igual à RQA da mesma
    matriz (0,9406); média do log do vão por nó, que é como a GNN vê o vão,
    0,8634. Na HVG: 0,9715 e 0,8869. Para referência, features de rede da
    Takens (`graph_takens`) 0,9211.
  - Redundância da vizinhança: correlação entre o valor do nó e a média dos
    vizinhos 0,97 (Takens) contra 0,91 (HVG); variância que resta após 4
    médias, 73% contra 47%; fração do grafo alcançada em 4 saltos, 44% contra
    5,5%.
  - Explicação proposta, compatível e não demonstrada: na rede de
    recorrência os vizinhos são pontos na mesma fase de outros batimentos, de
    amplitude quase igual, e a média sobre eles acrescenta pouco; a
    informação de ritmo está nos vãos, que a arquitetura reduz a uma média
    por nó. A HVG tem o mesmo gargalo nos vãos e a GNN funciona nela, então
    o gargalo só explica junto com a redundância.
  - Achado lateral, não usado em nenhum modelo da tabela: o histograma dos
    vãos da HVG sozinho dá 0,9715 no fold 8, acima de qualquer família
    isolada do chapman8. Surgiu post-hoc; exigiria protocolo próprio.
  - **Teste causal, registrado antes de rodar:** `train_line_gnn.py
    --span-bins 12` divide cada sentido em 12 relações por faixa logarítmica
    fixa do vão (1–1000 amostras), para o vão entrar em cada mensagem.
    Comparação no regime exploratório (treino 1–7, época escolhida no fold 8,
    teste fold 9), 3 sementes (42–44), contra o modelo original no mesmo
    regime (`--span-bins 0`), mesmos hiperparâmetros
    (`results/logs/run_posthoc_takens_gnn.sh`). Previsão: se a explicação
    estiver certa, B supera A no fold 9 de forma consistente entre sementes.
    Se não melhorar, a explicação dos vãos cai. Confundidor declarado: B tem
    0,91 M de parâmetros contra 0,54 M de A. Com --span-bins 0 a saída é
    idêntica à do modelo original (conferido) e os checkpoints antigos
    carregam.
- 15/nov/2026, **regularização aumentada nos dois braços do teste causal dos
  vãos, decidida pelo Eduardo. Não é post-hoc: o fold de teste não foi lido
  para tomar a decisão.** Na primeira rodada da semente 42, o braço B
  (`--span-bins 12`, 0,91 M de parâmetros contra 7.430 registros de treino)
  sobreajustou — perda de treino 0,0053 na época 39 com a validação parada em
  0,9654 desde a época ~13. A decisão usou **apenas o fold 8**, que é o
  conjunto de validação e existe para isso. Verificado no código antes de
  decidir: o laço de épocas só chama `prever` sobre `linhas["validation"]`, e
  o conjunto de teste aparece uma única vez (`train_line_gnn.py:371`), depois
  do laço e do `load_state_dict` do melhor checkpoint. O braço B foi morto na
  época 39 e **nunca leu o fold 9**.
  Mudança: `dropout` de 0,1 para 0,3 e `wd` de 1e-2 para 5e-2, **idênticos nos
  dois braços**, tudo o mais inalterado (hidden 64, 4 camadas, 50 épocas, lr
  1e-3, batch 32, sementes 42–44). Mudar só o B criaria um segundo confundidor
  e a comparação deixaria de isolar os vãos, que é a hipótese em teste; o
  confundidor já declarado (0,91 M contra 0,54 M de parâmetros) continua.
  O braço A da semente 42 com a configuração antiga chegou a rodar inteiro e
  leu o fold 9 uma vez (val 0,9380; teste 0,9323; época 42), e fica gravado em
  `posthoc_takens/gnn/takens_dir_d64_l4_e50_bins0_test9_val8_seed42.json`.
  Esse número **não** entra na comparação registrada, e não motivou a mudança:
  a motivação foi a curva de validação do B. A comparação válida é a nova, com
  os dois braços na regularização nova. Script atualizado em
  `results/logs/run_posthoc_takens_gnn.sh` (o anterior em `.sh.bak`); as saídas
  novas levam `do30_wd5e2` no nome. A previsão registrada não muda: se a
  explicação dos vãos estiver certa, B supera A no fold 9 de forma consistente
  entre as três sementes. O fold 10 do chapman8 não é lido em nenhum momento
  deste teste.
