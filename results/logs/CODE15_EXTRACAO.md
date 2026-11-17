# Máquina de 128 threads: extração e poda dos datasets externos

Divisão de trabalho, para as duas máquinas não escreverem nos mesmos arquivos:

| máquina | o que faz |
|---|---|
| `refri` (16 núcleos) | baixa as 18 partes do CODE-15%, descompacta e prepara (`ETAPA=download`); log em `results/logs/run_code15_download.log` |
| máquina de 128 threads | todo o resto: extração e poda, no Chapman agora e no CODE-15% quando o preparo terminar |

A fila da máquina grande está em `results/logs/run_maquina128.sh` e faz, em ordem:

1. **Chapman**: extrai as 8 famílias descartadas no PTB-XL (espectrograma,
   escalograma, GASF, GADF, MTF, RQA, Takens, transição) e roda a poda a
   partir de todas as famílias (`todas_familias`) no Chapman de 4 grupos e no
   de 8 ritmos. Estimativa: < 1 h.
2. **Espera** a `refri` terminar o preparo do CODE-15% (confere o log a cada 5 min).
3. **CODE-15%**: extrai as 15 famílias e o tsfresh e roda as podas
   `melhor_multiplex` e `tsfresh`. Estimativa: ~6 h.
4. **CODE-15%**: extrai as 8 famílias extras e roda a poda `todas_familias`.
   Estimativa: ~4 h.

Nenhuma etapa lê o fold 10. O ajuste final no teste (`evaluate_external.py`)
é feito depois, uma vez por conjunto (`results/external/PREREGISTRATION.md`).

## 1. Conferir o ambiente Python

O `.venv` do projeto aponta para `/usr/bin/python3.12` da `refri`. Teste na
máquina nova:

```bash
cd /exp/duducampos/InterpretECG
.venv/bin/python -c "import sys, numpy, pandas, sklearn, pycatch22, tsfresh, h5py, pyarrow; print(sys.version)"
```

Se funcionar, siga para o passo 2. Se der erro, crie um ambiente local à
máquina, fora do `/exp` compartilhado, com as mesmas versões da `refri`.
Não atualize nada: as features e o boosting precisam ser os mesmos do PTB-XL.

```bash
python3.12 -m venv ~/venv_interpretecg
~/venv_interpretecg/bin/pip install -r /exp/duducampos/InterpretECG/requirements.txt \
    h5py openpyxl numpy==2.5.4 pandas==3.1.0 scikit-learn==1.9.1 scipy==1.18.1 \
    pycatch22==1.0.2 tsfresh==0.21.2 joblib==1.6.0 threadpoolctl==3.7.0 pyarrow==26.0.0
```

e, no passo 2, acrescente `PY=~/venv_interpretecg/bin/python` antes do comando.

## 2. Rodar

Pode começar já: a etapa do Chapman não depende do CODE-15%.

```bash
cd /exp/duducampos/InterpretECG
NJOBS_EXTRACT=120 NJOBS_PRUNE=100 \
  setsid nohup results/logs/run_maquina128.sh > results/logs/run_maquina128.log 2>&1 < /dev/null &
```

`setsid nohup` mantém a fila rodando depois do logout.

Memória: com 120 processos, conte com ~30–40 GB livres na etapa do CODE-15%
(os sinais preparados ocupam ~6 GB). Se faltar, reduza `NJOBS_EXTRACT`.

## 3. Acompanhar

```bash
tail -5 results/logs/run_maquina128.log            # etapas concluídas, com data e hora
grep shard results/logs/extract_*_extra.log results/logs/extract_code15_full.log
grep "features  val\|escolhido" results/logs/prune_*todas_familias*.log
```

## 4. Se cair ou precisar parar

Todas as etapas retomam de onde pararam: a extração grava um checkpoint por
lote (`data/features/external/<dataset>/shards/`) e uma poda pronta não é
refeita. Basta rodar o comando do passo 2 de novo.

Para parar, mate pelo PID, nunca com `pkill -f` (o padrão casa com o próprio comando):

```bash
ps -eo pid,args | grep "[r]un_maquina128\|[e]xtract_external\|[p]rune_external"
kill <PID do run_maquina128.sh> <PID do python>
```

Na etapa do CODE-15% o script grava a trava `data/features/external/code15.LOCK`
com o nome da máquina. Se a máquina parar no meio e você quiser retomar em
outra, apague a trava antes.

## O que não fazer

- Não rodar `run_code15_full.sh` com `ETAPA=all` ou `ETAPA=extract` na `refri`:
  a extração do CODE-15% é da máquina grande.
- Não rodar `evaluate_external.py`: ele lê o fold 10, e cada conjunto é testado uma vez só.
