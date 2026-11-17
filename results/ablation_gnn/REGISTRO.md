# Registro: ablação HVG × L(HVG) da GNN

Gravado em 11/nov/2026, antes de qualquer execução da GNN sobre a HVG.
Decisão do Eduardo, na mesma data.

## Motivação

A GNN sobre o line graph L(HVG) foi escolhida porque as features sumarizadas
de L(HVG) em mV foram as melhores no exploratório. Esse critério é legítimo
(não usou teste), mas não atribui o desempenho: um revisor pode perguntar se a
mesma GNN sobre a própria HVG daria o mesmo, ou se passar mensagens no L(G) é
só um GNN de arestas. Os dois grafos são transformações determinísticas do
mesmo sinal — é o null de reparametrização do PRISM aplicado a um modelo
profundo.

Ficou de fora, por decisão discutida: a GNN no grafo caminho (só arestas
i–i+1). Com 4 camadas o campo receptivo seria de 9 amostras (90 ms) e ela
perderia por construção. A pergunta "a eficiência vem só de ser uma rede
pequena?" é respondida pela inception1d (507.653 parâmetros), já avaliada.

## Execuções (todas no regime exploratório)

Treino folds 1–7, escolha de época no fold 8, teste no fold 9. O fold 10 não
é lido. Configuração idêntica à exploratória do L(HVG): hidden 64, 4 camadas,
dropout 0,1, 50 épocas, batch 32, AdamW lr 1e-3, weight decay 1e-2, OneCycle.

| variante | grafo | parâmetros | sementes |
|---|---|---|---|
| HVG não direcionada (`--graph hvg`) | nós = amostras, arestas = visibilidade | 520.197 | 42, 43, 44 |
| HVG direcionada (`--graph hvg --directed`) | 2 relações: vizinho no passado / no futuro | 537.093 | 42, 43, 44 |
| L(HVG) não direcionada | já existente: 42; novas: 43, 44 | 520.453 | 42, 43, 44 |
| L(HVG) direcionada | já existentes: 42, 43; nova: 44 | 571.141 | 42, 43, 44 |

Atributos de nó na HVG: amplitude padronizada e posição t/1000; atributo de
mensagem: log do vão |j − i|. Camadas, leitura e cabeça idênticas à LineGNN
(`scripts/train_line_gnn.py`, classe `HVGGNN`).

## Análises previstas

- AUROC macro no fold 9, média ± dp das 3 sementes, por variante.
- Bootstrap pareado (1.000 reamostragens de registros do fold 9, AUROC média
  por semente): L(HVG) − HVG, separadamente para não direcionada e
  direcionada; direcionada − não direcionada dentro de cada grafo.
- Todos os resultados são reportados, inclusive se a HVG empatar ou vencer.
  Nesse caso o artigo diz que o ganho não é do line graph.

## Resultados (13/nov/2026, `scripts/bootstrap_gnn_ablation.py`, `bootstrap.json`)

Fold 9, AUROC macro, média ± dp de 3 sementes:

| | não direcionada | direcionada |
|---|---|---|
| HVG | 0,9147 ± 0,0006 | 0,9270 ± 0,0016 |
| L(HVG) | 0,9244 ± 0,0018 | 0,9281 ± 0,0002 |

Bootstrap pareado (1.000 reamostragens, IC 95%):
- L(HVG) − HVG, não direcionada: +0,0097 [+0,0062; +0,0131]
- L(HVG) − HVG, direcionada: +0,0010 [−0,0008; +0,0030]
- direcionada − não direcionada, HVG: +0,0123 [+0,0089; +0,0157]
- direcionada − não direcionada, L(HVG): +0,0036 [+0,0015; +0,0061]

Leitura: a seta do tempo responde pela maior parte do ganho. O line graph só
acrescenta quando a direção está ausente; com direção, HVG e L(HVG) empatam
(IC cruza zero). O artigo não pode atribuir o desempenho da GNN direcionada
ao line graph.
