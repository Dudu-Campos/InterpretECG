# Procedência

Os operadores deste subpacote foram copiados do projeto irmão **CNA**
(*Representational Adequacy*), que ainda não está publicado. A cópia é
deliberada: não é possível depender de um repositório não publicado.

- Origem: `/exp/duducampos/CNA/conversions/`
- Commit: `fb8499f1c56fd806c357671de3dc33e535efc8cb` (2026-08-09)
- Data da cópia: 2026-10-26
- Árvore de origem limpa no momento da cópia (`git status --porcelain` vazio)

## Mapa de arquivos

| Destino | Origem no CNA |
|---|---|
| `series_to_image/transforms.py` | `conversions/serie_to_image/series_to_image/transforms.py` |
| `series_to_image/markov.py` | `conversions/serie_to_image/markov.py` |
| `series_to_graph/visibility.py` | `conversions/series_to_graph/visibility.py` |
| `series_to_graph/transition.py` | `conversions/series_to_graph/transition.py` |
| `series_to_graph/takens.py` | `conversions/series_to_graph/takens.py` |
| `series_to_graph/features.py` | `conversions/series_to_graph/features.py` |
| `image_to_tabular/haralick.py` | `conversions/image_to_tabular/haralick.py` |
| `image_to_tabular/hog.py` | `conversions/image_to_tabular/hog.py` |
| `image_to_tabular/_validation.py` | `conversions/image_to_tabular/_validation.py` |

O aninhamento duplicado `serie_to_image/series_to_image/` da origem foi
normalizado para `series_to_image/`.

## O que NÃO foi copiado

O framework de conversão do CNA (`api.py`, `adapters.py`, `registry.py`, com
`RepresentationBatch`, `FeatureBatch` e `Converter`) ficou de fora. Este
projeto tem seu próprio pipeline de extração e cache (`features.py`,
`evaluate.py`), e os operadores são funções puras sobre `np.ndarray` e
`nx.Graph`. Também ficaram de fora as direções de conversão que o escopo do
artigo não usa: `tabular_to_image`, `tabular_to_graph`, `graph_to_series`,
`image_to_graph`, `image_to_series` e `graph_to_tabular`.

## Alterações feitas na cópia

1. **Remoção da dependência do framework em `image_to_tabular/`.** Foram
   truncadas as classes `HaralickImageTabularConverter` (origem, linhas
   329–402) e `HOGImageTabularConverter` (origem, linhas 131–227), além da
   função `validated_image_batch` de `_validation.py`. Eram as únicas
   referências a `conversions.api`. As funções por imagem
   (`haralick_features`, `hog_features` e auxiliares) estão intactas.
2. **Remoção da MTF duplicada de `transforms.py`** (ver seção abaixo), com
   o auxiliar `_aggregate_square`, que ficou órfão, e o ajuste da referência
   a Wang & Oates no docstring do módulo, já que a MTF saiu do arquivo.
3. **Adição de `_lphvg_edges` e do parâmetro `rho` em `visibility.py`.** É
   código novo, não veio do CNA. Ver seção "Adição: HVG limitada penetrável".
4. Nenhuma outra linha dos operadores foi alterada.

## Divergência na origem: duas MTF, uma removida

A origem tinha **duas** implementações de `markov_transition_field`:

- `markov.py` — estados por *midrank* preservando empates. Valores iguais
  sempre recebem o mesmo estado, por contrato documentado no próprio arquivo
  ("robust to tied quantiles").
- `transforms.py` — bordas por `np.quantile` com arestas internas
  deduplicadas via `np.unique` e `searchsorted(side="right")`.

**Elas não eram equivalentes.** Medido na derivação II do registro `00001_lr`
do PTB-XL (100 Hz), onde apenas 287 de 1.000 amostras são valores distintos —
71,3% são empates, porque o ADC quantiza o sinal:

| `n_bins` | diferença média abs. | diferença máx. abs. |
|---|---|---|
| 8 | 0,0133 | 0,3090 |
| 16 | 0,0126 | 0,4814 |

A escala da MTF é de probabilidade (0–1), então 0,48 é divergência de método,
não de arredondamento.

A de `transforms.py` foi **removida**, junto com seu auxiliar `_quantile_states`
e com `_aggregate_square`, que era usado só por ela. Fica a de `markov.py`, que
é a que o CNA expõe e a correta para ECG justamente porque empates dominam.

## Defeito herdado: `test_hvg_fbm` não testava nada

`test_hvg_fbm` veio do CNA **sem nenhuma asserção** — calculava o grafo,
imprimia e dava `return`. Passava sempre.

A afirmação do comentário original ("fBm deve ter caudas mais pesadas que iid,
logo `max_deg` maior") foi medida e **é falsa**: vale em 4 de 30 sementes.
Provavelmente foi por isso que a asserção nunca foi escrita.

Existe efeito robusto, mas em outra estatística: o grau médio do HVG de fBm
fica abaixo do de ruído i.i.d. em 30/30 sementes. A razão é estrutural, não de
tamanho finito: a HVG é outerplanar, logo `E <= 2N-3` e `<k> <= 4 - 6/N`, e o
ruído i.i.d. é quem **satura** esse teto. Séries com tendência caem abaixo —
uma série monótona dá um grafo caminho, com `<k> = 2`. Verificado em L=2000:

| série | `<k>` |
|---|---|
| monótona crescente | 1,999 |
| i.i.d. normal | 3,985 |
| senoide | 3,946 |
| passeio aleatório | 3,914 |
| *teto* `4 - 6/N` | *3,997* |

O teste foi **substituído** por três controles com asserção, ancorados no
resultado exato da LPHVG (seção seguinte), que é um alvo analítico muito mais
forte do que uma comparação qualitativa de caudas.

## Adição: HVG limitada penetrável (LPHVG)

`visibility.py` ganhou `_lphvg_edges` e o parâmetro `rho` em
`visibility_graph`. **Isto não veio do CNA**, é código novo.

Regra (Zhou et al. 2018, *Sci. Rep.* 8:5493): aresta `(i,j)` existe se no
máximo `rho` pontos intermediários obstruem a visibilidade horizontal, isto é
`#{k : i<k<j, x[k] >= min(x[i],x[j])} <= rho`. Com `rho=0` é a HVG comum.

A implementação usa um critério equivalente que permite parada antecipada:
sendo `M` o `(rho+1)`-ésimo maior valor entre os intermediários, a aresta
existe se `min(x[i],x[j]) > M`. Como `M` é não-decrescente em `j`, para um `i`
fixo basta parar quando `M >= x[i]` ou quando `M >= max(x[j+1..])`, usando
máximos de sufixo pré-computados. Sem a segunda parada, uma série decrescente
degeneraria para O(L³).

Validação:

1. `rho=0` reproduz `_hvg_edges` com igualdade exata de conjunto de arestas em
   seis tipos de série, incluindo empates, monótonas e constante.
2. Confere com a definição escrita ao pé da letra para `rho` de 0 a 3 em
   quatro tipos de série.
3. Para série i.i.d. o resultado exato é `<k> = 4*(rho+1)`, independente da
   distribuição de origem. Medido em 20 sementes com L=20.000: erro relativo
   de 0,05% em `rho=0` a 0,12% em `rho=3`. O desvio é sistemático, vem das
   bordas e cai como 1/L.

Custo em L=1000: 9,3 ms para `rho=1` em série i.i.d., contra 1,66 s do NVG.

**O que a LPHVG não resolve.** Medida em 200 registros do PTB-XL (derivação
II), a dispersão relativa entre registros praticamente não muda com `rho`:

| | `<k>` CV | expoente `C` CV |
|---|---|---|
| `rho=0` | 2,85% | 18,22% |
| `rho=1` | 2,35% | 18,11% |
| `rho=2` | 2,02% | 19,03% |

Ou seja, a penetrabilidade **não** aumenta o poder discriminativo desses dois
descritores. E o teto `<k> <= 4` preocupa menos do que parecia: em ECG real o
`<k>` da HVG comum varia em [3,380, 3,934], com só 10,5% dos registros acima de
3,9. O ganho da LPHVG aqui é outro: um alvo analítico exato para teste, e um
*null* de reparametrização limpo para o protocolo (`rho=0` contra `rho=1` é a
mesma visão com um único parâmetro mudado, e a teoria diz exatamente o que
deve mudar).

## Adição: quantificação de recorrência (RQA)

`image_to_tabular/rqa.py` **não veio do CNA**, é código novo, seguindo Marwan,
Romano, Thiel & Kurths (2007), *Physics Reports* 438(5-6):237–329.

A razão é um bug encontrado na primeira extração completa: a família
`image_recurrence` saiu com **100% das 336 colunas constantes**. O recurrence
plot limiarizado é binário com taxa de recorrência fixa em 5%, e a quantização
*equal-probability* de 256 níveis do Haralick colapsa essa imagem num único
nível de cinza (medido: `np.unique` do resultado é `[15]`). O GLCM fica
degenerado e as 14 medidas assumem valores fixos — `angular_second_moment` 1,0
e `contrast` 0,0 para todo registro.

Textura de nível de cinza é o readout errado para uma matriz binária. A seção
4.2 do protocolo já pedia o certo: taxa de recorrência, determinismo,
laminaridade e comprimento de diagonais. São nove medidas; a linha de
identidade e a banda de Theiler ficam fora das medidas diagonais.

Haralick continua sendo o readout das imagens de valor contínuo (campos
gramianos angulares, campo de transição de Markov, espectrograma, escalograma).

Validação (`tests/views/test_rqa.py`, 7 testes):

| série | determinismo | diagonal média |
|---|---|---|
| senoide periódica | 1,000 | 145,8 |
| ruído i.i.d. | 0,130 | 2,1 |
| passeio aleatório | 0,946 | 10,5 |

Em 30 registros reais do PTB-XL (derivação II), oito das nove medidas variam
entre registros, com CV de 6,5% (laminaridade) a 59% (comprimento vertical
máximo). A nona, `recurrence_rate`, é **constante por construção**, já que a
taxa é fixada em 0,05 — ela é descartada na família, como as colunas de peso
dos grafos não-ponderados, para que a largura declarada seja igual à efetiva.
A família `image_recurrence` tem portanto 8 features por derivação, não 9.

Custo: ~190 ms por derivação, contra ~30 ms do Haralick, por causa dos laços
sobre as 940 diagonais e colunas.

## Defeito herdado: dimensão de correlação com limiar de recorrência nulo

A primeira extração completa das 12 derivações quebrou em 58% (ECG 12722),
em `C_takens_correlation_dim`, com `ValueError: Geometric sequence cannot
include zero`.

A causa é o bracket de raios do ajuste de Grassberger-Procaccia:

```python
rs = np.geomspace(eps / 10, eps * 1.5, n_r)
```

`eps` é o limiar `epsilon` escolhido por `takens_recurrence_graph` como o
quantil 0,05 das distâncias entre vetores de embedding. Esse quantil é **zero**
quando mais de 5% dos pares de vetores são exatamente idênticos, o que acontece
em derivações com trechos longos de sinal achatado. O grafo em si continua bem
definido — `query_pairs(r=0)` devolve os pares de distância nula —, mas o
bracket perde escala.

O código tinha um fallback para `eps is None or not np.isfinite(eps)` que não
cobria `eps == 0`. Agora cobre, e há um segundo guarda: se o bracket derivado da
própria nuvem de pontos também for nulo, `D2` é **indefinida**, não apenas
desconhecida, e a função devolve NaN. Dois testes em
`tests/views/test_graph_features.py` fixam os dois caminhos.

### O que isso diz sobre o PTB-XL

Varredura das 261.588 combinações registro × derivação (desvio-padrão e número
de valores distintos por derivação):

| valores distintos | derivações | fração |
|---|---|---|
| ≤ 1 | 1 | 0,0004% |
| ≤ 30 | 2 | 0,001% |
| ≤ 100 | 8 | 0,003% |

Mediana: 330 valores distintos em 1.000 amostras. Nenhuma amostra NaN em todo
o dataset.

O caso único é o **registro 12722**: V5 é identicamente zero (eletrodo morto) e
V6 tem 26 valores distintos. Não é tratado como registro inválido — os rótulos
são válidos e as features degeneradas são descritivas, não ausentes. A RQA de
V5, por exemplo, devolve determinismo 1,0 e laminaridade 1,0, que é a descrição
correta de uma matriz de recorrência cheia. O único NaN resultante é o `C` de
V5 nas três famílias de grafo.

Note que a condição que quebrava **não é** "derivação constante": V6 tem
amplitude 1,125 e também zerava o limiar. Um guarda no nível da derivação, do
tipo `ptp(x) == 0`, teria passado por V6 — por isso a correção é na raiz.

## Guarda contra cache de readout antigo

`ViewExtractor.extract` é incremental, e a troca Haralick → RQA mudou os nomes
das colunas de `image_recurrence`. Um `pd.concat` entre o cache antigo (28
colunas por derivação) e as linhas novas (8) teria feito a **união** dos dois
esquemas, deixando as duas metades pela metade em NaN, sem erro. O `flush`
agora compara os conjuntos de colunas e recusa a mistura. O manifesto já
validava as derivações; passou a haver verificação equivalente para o readout.

## Adição: parâmetros fixos de treino (famílias `_abs`, 02/nov/2026)

Todos os operadores copiados normalizavam por amostra e, com isso, descartavam
a amplitude da derivação: min-max na GAF, quantis da própria série na MTF e no
grafo de transição, taxa de recorrência fixa no RP e na rede de Takens, e
quantização equal-probability por imagem no Haralick. Foi medido que isso
custa caro: 60 estatísticas de amplitude em mV somam +0,026 de AUROC ao
catch22, e +0,087 em HYP.

Cada operador ganhou um parâmetro opcional que troca a normalização por uma
constante fixa. Sem o parâmetro, o comportamento é o de antes, e as famílias
originais não mudam:

| operador | parâmetro novo |
|---|---|
| `gramian_angular_field` | `scale`: divide por ela e corta em ±1 |
| `markov_transition_field` | `bin_edges`: bordas internas fixas dos estados |
| `transition_graph` | `bin_edges`: idem |
| `quantize_gray_image`, `haralick_features` | `thresholds`: cortes de cinza fixos |
| `recurrence_plot`, `takens_recurrence_graph` | já aceitavam `epsilon` |

As constantes vêm de `scripts/calibrate_absolute_views.py`, só dos folds 1–7,
e ficam em `data/features/absolute_calibration.json`. As famílias resultantes
têm o sufixo `_abs`, ficam em `data/features/views_abs/` e são registradas em
`FAMILY_OPERATORS`, fora de `DEFAULT_FAMILIES`.

Na `image_recurrence_abs`, a taxa de recorrência deixa de ser constante e
volta a ser feature (9 medidas, não 8). A imagem e o grafo de recorrência
usam o mesmo ε e continuam lendo a mesma matriz: conferido em 6 derivações,
com contagem de arestas idêntica.

Os visibility graphs (HVG e LPHVG) não têm variante, porque dependem só da
ordem das amostras.
