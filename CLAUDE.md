# Projeto CBMS — Múltiplas visões de ECG e interpretabilidade

Artigo-alvo: IEEE CBMS 2027 (6 páginas, formato IEEE de duas colunas, extensível a 8 páginas pagas).
Autor: Eduardo de Moura Campos (ICMC-USP), orientação de Alneu de Andrade Lopes.
Projeto irmão: PRISM (classificação de estilo pictórico com múltiplas visões handcrafted, em revisão na *Pattern Recognition*). Este projeto reaproveita a **lógica experimental** do PRISM (controles placebo e null, matriz de redundância, propagação de rótulos), não o código de features visuais.

---

## 1. Ideia central

Um mesmo sinal de ECG pode ser re-representado como **série temporal**, como **imagem** e como **grafo**. Como essas visões são transformações determinísticas do mesmo sinal, nenhuma contém informação nova no sentido de Shannon (desigualdade de processamento de dados). O que muda entre elas é a **acessibilidade**: quais padrões viram features simples, separáveis e nomeáveis em cada representação. Trata-se, portanto, de uma pergunta sobre **viés indutivo de representações**.

**Pergunta de pesquisa:** representações estruturalmente distintas do mesmo sinal tornam explícitos padrões clinicamente relevantes diferentes? Essa complementaridade pode ser medida, e não apenas afirmada?

**Lacuna na literatura:** trabalhos com visibility graphs, GAF, recurrence plots e fusão multi-view em ECG são orientados a acurácia. Eles *afirmam* complementaridade para justificar arquiteturas de fusão, mas não a *medem*, não usam controles e não verificam alinhamento com marcadores clínicos.

**Posicionamento contra o vizinho mais próximo** (ECG-IMN, CBMS 2026, arXiv:2602.09566): lá, a interpretabilidade é imposta pela arquitetura (hiper-rede que gera um modelo linear por amostra) e opera sobre amostras cruas. Aqui, ela vem de **representações cujas features já têm semântica** e é **verificada contra marcadores clínicos**.

---

## 2. Escopo

- **Fonte:** apenas séries temporais (ECG). Imagens médicas como fonte ficam para trabalho futuro.
- **Visões derivadas:** série, imagem e grafo (seção 4).
- **Dataset principal:** PTB-XL (versão de 100 Hz) com PTB-XL+ para os marcadores clínicos.
- **Generalização (opcional, só se sobrar tempo):** tabela compacta em 1–3 datasets extras (Chapman-Shaoxing, CODE-15%, PhysioNet/CinC 2017), com 1–2 números-resumo por dataset e sem refazer figuras.
- **Abordagem:** features handcrafted + classificadores rasos, para que a interpretação por feature seja possível.

### Fora de escopo
- Fusão aprendida, *joint loss* e arquiteturas multi-view profundas.
- Delineação própria de ondas P-QRS-T (usar os marcadores do PTB-XL+).

---

## 3. Dados

### PTB-XL
- Cerca de 21,8 mil registros de 12 derivações, 10 s, de cerca de 18,9 mil pacientes (PhysioNet, v1.0.3).
- Usar a versão de **100 Hz** (1.000 amostras por derivação) por custo computacional.
- **Folds oficiais (`strat_fold`):** 1–8 treino, 9 validação, 10 teste. Os folds são por paciente; **nunca** reembaralhar registros entre folds, para evitar vazamento entre pacientes. Este projeto *não* usa esse recorte na exploração: o fold 10 é selado (seção 6.1).
- **Rótulos:** declarações SCP-ECG. Recorte de classes é decisão pendente (seção 8).
- **Qualidade do sinal:** medida nas 261.588 combinações registro × derivação — nenhuma amostra NaN, mediana de 330 valores distintos em 1.000 amostras. Uma única derivação é degenerada em todo o dataset: **V5 do registro 12722** é identicamente zero (eletrodo morto), e V6 do mesmo registro tem 26 valores distintos. O registro é mantido, com as features degeneradas tratadas como descritivas (ver `src/interpretecg/views/PROVENANCE.md`). Confirmado por varredura independente: 1 combinação com amplitude zero, 2 com menos de 50 valores distintos, mediana de 330.
  Cuidado com o critério de detecção: o limiar de recorrência também zerou em **V4** do mesmo registro, que tem 354 valores distintos e amplitude de 4,585 mV. A causa é que platôs longos geram vetores de embedding idênticos, então um filtro por diversidade de valores ou por `ptp == 0` **não** pega esse caso. O PTB-XL anota o problema: `electrodes_problems = 'v4, v5'` para o registro 12722.
- **Anotações de qualidade subutilizadas:** `ptbxl_database.csv` traz `static_noise` (3.260 registros, 15,0%), `baseline_drift` (1.598, 7,3%), `burst_noise` (613, 2,8%) e `electrodes_problems` (30, 0,1%). O pipeline hoje as ignora. Vale ao menos um teste de sensibilidade excluindo os 30 com defeito de eletrodo.
- O arquivo `RECORDS` publicado tem um defeito upstream na linha 21799 e perde o registro 21837. A lista canônica é a coluna `filename_lr` do `ptbxl_database.csv`.

### PTB-XL+
- Features clínicas pré-computadas por registro (intervalos, durações, amplitudes, batimentos medianos), extraídas por mais de um algoritmo (comerciais e abertos).
- Uso: **verdade-terreno de marcadores clínicos** para testar o alinhamento das features de cada visão (seção 6.4).
- Junção com o PTB-XL por `ecg_id`.

---

## 4. Visões e famílias de features

Cada visão é uma **família**, no sentido do PRISM. Os operadores vivem em
`src/interpretecg/views/`, copiados do projeto irmão CNA e documentados em
`src/interpretecg/views/PROVENANCE.md`.

**Decisão tomada (2026-10-26):** as visões de grafo e imagem são calculadas nas
**12 derivações**, não num par. Uma correção de desempenho tornou isso viável
(materializar a componente conexa antes do BFS em `_path_stats` deu 8,7×), e
com 96 processos a extração completa leva cerca de 1,5 h. Extrair as 12 elimina
a necessidade de justificar uma escolha de derivação no artigo e torna a
comparação com o baseline Catch22, que usa as 12, direta.

Uma família por operador, um Parquet cada, em `data/features/views/`: seis de
imagem e quatro de grafo. A série de intervalos RR permanece não implementada
(seção 4.1).

### 4.1 Série temporal (nativa)
- Estatísticas no domínio do tempo e espectrais por derivação.
- Features de HRV a partir da série RR (detecção de picos R via `neurokit2`).

### 4.2 Imagem (derivada)
- **Recurrence plot (RP)**, com quantificação de recorrência: determinismo,
  laminaridade, comprimento médio e máximo de diagonais, tempo de aprisionamento,
  entropia de diagonais, divergência. Oito medidas; a taxa de recorrência é
  fixada em 0,05 e portanto não é feature.
- **Gramian Angular Field** (GASF e GADF) e **Markov Transition Field**, com
  textura de Haralick (28 medidas: as 14 clássicas em média e amplitude angular).
- **Espectrograma** (STFT) e **escalograma** (CWT de Morlet), também com Haralick.
- Implementação própria em `views/series_to_image/` e `views/image_to_tabular/`;
  `pyts` não é usado.

**Textura não serve para o RP.** O RP limiarizado é binário, e a quantização
equal-probability do Haralick o colapsa num único nível de cinza: as 14 medidas
ficam constantes em todo registro. Foi medido — 336 de 336 colunas com um único
valor distinto em 12.000 registros. O readout do RP é RQA (Marwan et al. 2007);
Haralick fica para as imagens de valor contínuo.

### 4.3 Grafo (derivado)
- **Visibility graph horizontal** (Luque et al., 2009) e sua variante
  **limitada penetrável** com ρ=1 (Zhou et al., 2018). A LPHVG tem resultado
  exato para série i.i.d., ⟨k⟩ = 4(ρ+1), verificado com erro de 0,05% a 0,12% —
  é o controle positivo analítico da família e um *null* de reparametrização
  com previsão fechada. O visibility graph **natural** foi deixado de fora: é
  180× mais caro que a LPHVG (1,66 s contra 9,3 ms por derivação) e pode ser
  acrescentado depois, porque o cache é por família.
- **Rede de recorrência de Takens**: adjacência do RP sem a diagonal, do espaço
  de fase reconstruído por embedding.
- **Grafo de transição por quantis**, o único ponderado, logo o único cujas duas
  features de peso não são NaN.
- Descritores: 26 comuns mais um escalar diagnóstico por conversão (expoente de
  grau, taxa de entropia, dimensão de correlação), de propósito com a mesma
  dimensionalidade entre conversões. `n_components` e `largest_cc_frac` são
  descartados nos grafos de visibilidade, onde a conectividade é estrutural —
  todo gerador emite a aresta (i, i+1). Implementação própria em
  `views/series_to_graph/`; `ts2vg` e `pyunicorn` não são usados.

### 4.4 Experimento-chave: RP-como-imagem vs. RP-como-rede
A matriz de recorrência é **o mesmo objeto** lido de duas formas: como imagem
(RQA) e como grafo (medidas de rede). Isso é um experimento natural sobre viés
indutivo, porque o dado é idêntico e só muda a leitura. Comparar redundância,
desempenho e alinhamento clínico entre as duas leituras.

Para que isso seja verdade, as duas famílias usam **parâmetros casados**: o
mesmo embedding congelado, a mesma taxa de recorrência (0,05) e a mesma janela
de Theiler (1). Verificado: a imagem e a adjacência do grafo divergem em 0 ou 1
aresta de cerca de 440 mil pares, ou seja 0,000%. Sem esse casamento o
experimento não existiria — os padrões de biblioteca dão `dimension=1,
delay=1, rr=0,10` para a imagem e embedding automático com `rr=0,05` para o
grafo, que são matrizes diferentes.

### 4.5 Line graph da HVG (achados de 07/nov/2026)

Implementação em `views/series_to_graph/line.py`, extração em
`scripts/extract_line_graph_features.py`. Um nó de L(G) é uma aresta (i, j)
da HVG — uma "corda" horizontal no nível min(xᵢ, xⱼ), com degrau |xᵢ − xⱼ| e
vão j − i; duas cordas se ligam quando compartilham uma amostra (cunha).
Forma fechada, sem construir L(G). Quatro famílias: `graph_{hvg,lphvg1}_line`
(28 features ordinais por derivação, invariantes como a HVG) e
`graph_{hvg,lphvg1}_line_mv` (21 em mV, não invariantes).

**Desempenho (boosting de configuração fixa, val fold 8 / teste fold 9):**
- L(HVG) mV sozinho, 252 features: 0,9119 / 0,9006 — acima do catch22
  (0,8927 / 0,8768) e da família de amplitude (0,8676 / 0,8568). Primeira
  visão derivada que, isolada, supera a série.
- A parte ordinal soma pouco sobre a HVG (+0,007); o ganho está na parte mV.
- Base (catch22 + amplitude + interderivação) + visibilidade + line graph + mV:
  **0,9280 / 0,9179**, melhor validação até aqui. CNN: 0,9436 / 0,9287.
- Em pacientes (fold 9, especificidade fixa em 90%): a CNN detecta 27 MI e
  37 CD a mais; empatamos em STTC e superamos em HYP (192 contra 187 de 268).

**Interpretabilidade (importância por permutação em grupo, fold 8,
`scripts/line_graph_importance.py`):** quase toda a importância está em
poucas features mV de nível das cordas, cada classe com assinatura própria,
nas derivações que a clínica usa:
- HYP ← `level_std` (queda 0,128), em V5 e V1 — critério de Sokolow-Lyon.
  `level_std` em V5 tem ρ = +0,78 com a amplitude pico a pico do QRS em V5
  (12SL do PTB-XL+); em V1, +0,80.
- STTC ← `level_q90`, em aVR, V5, V6, I. `level_q90` em V5 tem ρ = +0,67 com
  a amplitude da onda T.
- CD ← `span_step_corr`, em V1. Em V1 tem ρ = +0,65 com a amplitude de R
  (R′ do bloqueio de ramo direito).
- MI ← derivações III, V2, aVL, II, V3 (inferior, anterosseptal, recíproca).
- As 28 features ordinais têm importância ≤ 0,002 cada.

Ressalvas ainda abertas: importância por derivação não controla a redundância
das derivações aritméticas (seção 6.4); o fold 8 também serviu ao early
stopping; correlação com o marcador não prova que o modelo o usa; falta
comparar a concentração com catch22 e amplitude antes de afirmar que o line
graph é *mais* interpretável.

**Seta do tempo:** as 49 features do line graph (e as da HVG) são exatamente
invariantes à inversão temporal — verificado, zero diferenças entre x e
x[::-1]. A HVG não distingue passado de futuro, e o ECG é fortemente
irreversível (QRS sobe rápido, T desce devagar; P antes do QRS, ST depois).
Candidata a próxima família: HVG/L(G) direcionados (grau para passado vs.
futuro, irreversibilidade de Lacasa et al. 2012; cunhas em cadeia, leque para
frente e leque para trás; degrau com sinal) — um *null* controlado de "ler ou
não a seta do tempo" sobre o mesmo grafo.

**GNN sobre L(HVG)** (`scripts/train_line_gnn.py`, cache em
`scripts/build_hvg_cache.py`): passagem de mensagens por média, não
direcionada, pesos compartilhados entre derivações, regime exploratório.
Primeira rodada em andamento; na época 11 de 40, val 0,9182 (acima da
sumarização mV do mesmo grafo). É modelo profundo — fora do escopo da seção 2
se virar contribuição; por ora é teste.

---

## 5. Hipóteses

- **H1:** as visões são complementares. A redundância intra-família supera a inter-família, e combinar famílias distintas supera reparametrizações dentro de uma família.
- **H2:** visões diferentes alinham com marcadores clínicos diferentes. Por exemplo, a visão de grafo concentra irregularidade de ritmo (variabilidade RR, FA), enquanto séries e imagens concentram morfologia (duração de QRS, alterações de ST-T).
- **H3:** ler a mesma matriz de recorrência como imagem ou como grafo produz features com padrões distintos de redundância e de alinhamento clínico.

---

## 6. Protocolo experimental

### 6.1 Avaliação
- Classificadores rasos e interpretáveis (regressão logística regularizada, random forest e/ou gradient boosting).
- **Avaliação indutiva** nos folds oficiais. A seleção de features e de hiperparâmetros usa apenas treino e validação.
- Métrica principal: AUROC macro (multirrótulo) ou acurácia balanceada, conforme o recorte de classes.

**O fold 10 é selado.** Toda a análise exploratória — visões isoladas,
combinações, escolha de derivações e de parâmetros, placebo e nulls de
reparametrização — roda no regime `exploratory`, que usa folds 1–7 para treino,
fold 8 para validação e fold 9 como teste exploratório. O fold 10 só é lido no
regime `confirmatory` (folds 1–8 treino, 9 validação, 10 teste), que é o
protocolo oficial do PTB-XL e o único número comparável a Strodthoff et al.

A motivação é inferência seletiva: o protocolo desta seção prevê dezenas de
avaliações, e repeti-las no fold 10 transformaria a estimativa final em
otimismo de seleção. O CLI recusa o regime confirmatório sem uma justificativa
explícita e registra cada uso em `results/test_fold_usage.jsonl`, de modo que o
artigo possa declarar quantas vezes o conjunto de teste foi consultado.

**Decisão do Eduardo em 26/out/2026: o fold 10 não é tocado até o fim.** Não se
gasta uso confirmatório para comparar números parciais com o benchmark, nem
para checar se um resultado exploratório "se mantém". A rodada confirmatória
acontece uma única vez, com a configuração final já escolhida e congelada.

**Exceção, decidida pelo Eduardo em 29/out/2026: concorrentes de configuração
fixa.** Um baseline cuja configuração é a publicada e congelada — nenhum
hiperparâmetro, época ou limiar escolhido olhando validação ou teste — não
sofre de otimismo de seleção, então pode ser avaliado no fold 10 uma vez. Vale
para a xresnet1d101 (`scripts/train_cnn.py`, protocolo do Strodthoff sem early
stopping), que antes foi validada no regime exploratório (fold 9: 0,9287,
contra 0,928 publicado no fold 10). Continua obrigatório o registro em
`results/test_fold_usage.jsonl`. O risco que sobra, e que o artigo deve
declarar, é indireto: saber a distância para a CNN no fold 10 pode orientar
escolhas do nosso método. Decisões de método se apoiam no número exploratório
da CNN, não no confirmatório. A exceção **não** vale para nada que seja nosso.

Consequência prática ao escrever: enquanto isso, nenhum número nosso é
comparável aos de Strodthoff et al., porque o protocolo difere em duas coisas
ao mesmo tempo — fold 9 contra fold 10, e treino em 1–7 contra 1–8 (12,5%
menos dados). Toda tabela que puser os dois lado a lado antes da rodada final
precisa dizer isso. Indício de que a diferença é pequena, não medição: no
catch22 a passagem exploratório→confirmatório moveu apenas +0,003.

Referência exploratória do baseline Catch22 (264 features, 12 derivações):
AUROC macro de 0,8425 no fold 9 com regressão logística. A execução
confirmatória correspondente deu 0,8456 no fold 10 — diferença de 0,003 apesar
de 12,5% menos dados de treino, o que indica que o holdout exploratório ordena
comparações de forma confiável. Diferenças por classe são maiores (HYP varia
0,04), então afirmações por classe exigem cautela.

**A escolha de classificador move o resultado mais do que a regularização.**
Nas mesmas 264 features e no mesmo fold 9: logística 0,8425, random forest
0,8613, histogram gradient boosting 0,8726. São +0,0301 de AUROC, ou 19% menos
erro de ranqueamento, só trocando a classe de modelo. A grade de `C` da
logística, que varia apenas na quarta decimal em quatro ordens de magnitude,
mostra que o modelo não está limitado por *overfitting* — mas não licencia a
conclusão de que a representação é o gargalo, porque `C` não sai da família
linear. Esse erro de inferência foi cometido e corrigido; o registro está em
`results/RESULTS.md`, experimentos 1 e 3.

Consequência para o protocolo: **reportar os dois classificadores por família.**
A diferença entre o gradient boosting e a logística, por família, mede quanto da
informação daquela visão está codificada de forma não-linear — o que é uma
medida de acessibilidade por si, e encaixa na H1. Comparar visões apenas com um
modelo linear mede quanto cada visão já vem linearmente separável, que é uma
afirmação mais estreita e precisa ser dita assim.

### 6.2 Comparações
- Cada visão isolada × combinações de visões × todas as visões.
- **Baseline profundo único**, só como referência de patamar: CNN 1D na série crua (por exemplo, xresnet1d do benchmark de Strodthoff et al. sobre o PTB-XL).

### 6.3 Controles (herdados do PRISM)
- **Placebo:** features aleatórias com a mesma dimensionalidade de uma família. Devem degradar ou não acrescentar desempenho.
- **Null de reparametrização:** variantes de uma mesma visão tratadas como "nova família" (GASF vs. GADF, parâmetros de embedding diferentes para a rede de recorrência). Se a hipótese da diversidade vale, essas variantes devem se comportar como o placebo.

### 6.4 Medidas de complementaridade e interpretabilidade
- **Matriz de redundância** inter e intra-visão (correlação absoluta média ou informação mútua entre features), com a razão intra/inter.
- **Propagação de rótulos de família:** se uma feature migra para outra família no espaço de redundância, a visão não é coesa.
- **Alinhamento clínico:** correlação e informação mútua entre as features mais importantes de cada visão e os marcadores do PTB-XL+. Visões interpretáveis devem concentrar um marcador em poucas features.
- **Casos discordantes:** registros que uma visão acerta e outra erra, caracterizados pelos marcadores clínicos.

**Calibrador de graça: a redundância entre derivações é conhecida.** As 12
derivações são 8 dimensões. Um registro é uma matriz 1000 × 12 de posto
numérico 8 (medido: 35 de 40 registros dão exatamente 8), e três componentes
explicam 97,7% da energia — é o dipolo cardíaco, um vetor em 3D projetado em
eixos diferentes. Quatro derivações são aritmética pura, não medição:

```
III = II − I      aVR = −(I + II)/2      aVL = I − II/2      aVF = II − I/2
```

Verificado em 50 registros, exato até o arredondamento do ADC; a lei de
Einthoven (I + III = II) vale com erro relativo de 1,4×10⁻³. Reconstruindo cada
derivação a partir de I e II por mínimos quadrados, há um **gradiente de
redundância com resposta conhecida**: R² = 1,000 para as seis dos membros,
0,804 para V6, 0,637 para V4, 0,364 para V2. As precordiais não são redundantes
porque ficam perto do coração, onde a aproximação de dipolo quebra e aparece
informação local não-dipolar — que é o que localiza o território do infarto.

A matriz de redundância **tem** que recuperar esse gradiente. Se não marcar
III, aVR, aVL e aVF como redundantes com I e II, a medida está quebrada, e isso
é verificável sem rótulo nenhum.

E há um achado nesse calibrador: dependência linear **exata** no sinal produz
correlação de apenas 0,42 a 0,71 entre as mesmas features do catch22 em
derivações dependentes. A informação é provadamente idêntica; o que muda é a
acessibilidade. É o fenômeno central do artigo observado num caso de verdade
conhecida, e serve de exemplo calibrador na seção 6.4.

---

## 7. Restrições práticas

- Redes de recorrência e RPs são O(n²) em memória. Com 1.000 amostras são 10⁶ entradas por derivação, o que é tratável, mas não deve ser feito para 12 derivações × 21 mil registros sem cache. Calcular por lote e salvar as features em disco (parquet).
- Fixar sementes e registrar os parâmetros de cada transformação (embedding, limiar de recorrência, resolução das imagens).
- Pipeline em etapas cacheáveis: sinal → visões → features → avaliação.

---

## 8. Decisões pendentes

1. **Recorte de classes:** as 5 superclasses diagnósticas (NORM, MI, STTC, CD, HYP) ou um subconjunto semântico que contraste ritmo (FA, onde o grafo deve se destacar) com condução (bloqueios de ramo, onde a morfologia deve dominar). Os rótulos de ritmo do PTB-XL estão em declarações separadas das superclasses diagnósticas.
2. ~~**Derivações**~~ — **resolvida (2026-10-26):** as 12. Ver seção 4. Fica
   em aberto, mas como escolha de análise e não de extração, se a tabela
   principal usa as 12 ou apenas as 8 independentes (I, II, V1–V6); as 4
   aritméticas inflam a contagem de features sem acrescentar medição.
3. ~~**Parâmetros de embedding e limiar**~~ — **resolvida (2026-10-26):**
   congelados no nível do dataset, em `data/features/takens_embedding.json`.
   Estimados por `auto_tau_mi` e `auto_m_fnn` sobre 400 registros amostrados
   dos folds 1–7 (regime exploratório), mediana por derivação: τ entre 9 e 12
   (90–120 ms), m entre 7 e 12. Taxa de recorrência 0,05 e janela de Theiler 1,
   compartilhadas com a imagem de recorrência (seção 4.4).
   Nota: o `m_max` padrão da implementação é 10 e a função **retorna o teto**
   quando não converge, o que censurava até 52% das estimativas em V2. Com
   `m_max=25` a censura máxima cai a 2,5%.
4. **Datasets de generalização:** se e quais.

---

## 9. Referências-base

- Wagner et al. (2020), *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data.
- Strodthoff et al. (2023), *PTB-XL+, a comprehensive electrocardiographic feature dataset*, Scientific Data.
- Lacasa et al. (2008), *From time series to complex networks: the visibility graph*, PNAS.
- Marwan et al. (2009), *Complex network approach for recurrence analysis of time series*, Physics Letters A.
- Wang & Oates (2015), *Imaging time-series to improve classification and imputation*, IJCAI.
- Kachhara & Ambika (2019), *Bimodality and scaling in recurrence networks from ECG data*, EPL.
- Kutluana & Turker (2024), visibility graphs no PTB-XL, Biomedical Signal Processing and Control.
- Thambawita et al. (2026), *ECG-IMN*, CBMS 2026 (arXiv:2602.09566). Principal vizinho para posicionamento.
