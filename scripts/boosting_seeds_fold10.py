"""Sementes 42-46 do ajuste final dos boosting (catch22, tsfresh, nosso) no fold 10.

Listas de features fixas (as mesmas das execucoes originais); so a semente do
HistGradientBoosting muda (sorteio de features por divisao, max_features=0.5).
Mesmo criterio dos modelos profundos: 5 sementes, media +- dp. A semente 42 e
reajustada e conferida contra o numero ja gravado (reprodutibilidade).
Decisao do Eduardo em 13/nov/2026; cada leitura vai para test_fold_usage.jsonl.

Registros e listas por dataset, identicos aos originais:
- ptbxl: catch22 como eval_catch22_fold10.py; tsfresh e melhor_multiplex como
  confirm_boosting.py (listas congeladas, intersecao de registros).
- chapman8, code15: catch22 como eval_catch22_fold10.py; tsfresh e a lista
  "nossa" (todas_familias no chapman8 e, desde 15/nov/2026, no code15; antes
  melhor_multiplex no code15, chave code15/melhor_multiplex) como
  evaluate_external.py --lists pruned.

Saida: <dir>/seeds/boosting_<conjunto>_seed<s>_scores.npz e
results/boosting_seeds_fold10.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import PTBXL, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

SEMENTES = (42, 43, 44, 45, 46)
NOSSO = {"ptbxl": "melhor_multiplex", "chapman8": "todas_familias", "code15": "todas_familias"}
MOTIVO = ("semente {s} do ajuste final, lista de features fixa, configuracao fixa, early stopping no fold 9; "
          "media +- dp de 5 sementes como nos modelos profundos; decisao do Eduardo 13/nov/2026")


def partes(nomes):
    p = {}
    for f in nomes:
        fam, col = f.split("::", 1)
        p.setdefault(fam, []).append(col)
    return p


def montar(quadros, nomes, ids):
    x = {n: np.hstack([np.array(quadros[fam].loc[ids[n], cs].to_numpy(np.float32), copy=True)
                       for fam, cs in partes(nomes).items()]) for n in ids}
    for a in x.values():
        a[np.isinf(a)] = np.nan
    return x


def folds(fold: pd.Series):
    return {"train": fold.index[fold.between(1, 8)].to_numpy(), "validation": fold.index[fold.eq(9)].to_numpy(),
            "test": fold.index[fold.eq(10)].to_numpy()}


def problemas(ds):
    """(conjunto, x, y, rotulos, auc gravada da semente 42, dir de saida) por conjunto."""
    if ds == "ptbxl":
        out = ROOT / "results/confirmatory"
        reg = PTBXLDataLoader(PTBXL).records.set_index("ecg_id")
        c22 = pd.read_parquet(ROOT / "data/features/catch22_12leads.parquet").set_index("ecg_id")
        ids = folds(reg["strat_fold"])
        rot = list(SUPERCLASSES)
        nomes = [f"catch22::{c}" for c in c22.columns]
        yield ("catch22", montar({"catch22": c22}, nomes, ids), {n: reg.loc[v, rot].to_numpy(np.int8) for n, v in ids.items()},
               rot, json.load(open(out / "boosting_catch22.json"))["test_macro_auroc"], out)
        listas = json.load(open(out / "frozen_feature_lists.json"))
        quadros = carregar_quadros()
        quadros["tsfresh"] = pd.read_parquet(ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
        fams = {f.split("::")[0] for v in listas.values() for f in v["features"]} | {"catch22"}
        comuns = None
        for fam in sorted(fams):
            comuns = quadros[fam].index if comuns is None else comuns.intersection(quadros[fam].index)
        r = PTBXLDataLoader(PTBXL).records
        r = r.loc[r.ecg_id.isin(comuns)].reset_index(drop=True)
        masc = PTBXLDataLoader.split_masks(r, "confirmatory")
        ids = {n: r.loc[m, "ecg_id"].to_numpy() for n, m in masc.items()}
        rr = r.set_index("ecg_id")[rot]
        y = {n: rr.loc[ids[n]].to_numpy(np.int8) for n in ids}
        for nome in ("tsfresh", "melhor_multiplex"):
            yield (nome, montar(quadros, listas[nome]["features"], ids), y, rot,
                   json.load(open(out / f"boosting_{nome}.json"))["test_macro_auroc"], out)
        return
    out = ROOT / "results/external" / ds
    lab = pd.read_csv(ROOT / "data/external" / ds / "labels.csv").set_index("ecg_id")
    rot = json.load(open(ROOT / "data/external" / ds / "prep.json"))["labels"]
    feats = ROOT / "data/features/external" / ds
    ids = folds(lab["strat_fold"])
    y = {n: lab.loc[v, rot].to_numpy(np.int8) for n, v in ids.items()}
    c22 = pd.read_parquet(feats / "catch22.parquet").set_index("ecg_id")
    yield ("catch22", montar({"catch22": c22}, [f"catch22::{c}" for c in c22.columns], ids), y, rot,
           json.load(open(out / "boosting_catch22.json"))["test_macro_auroc"], out)
    for nome in ("tsfresh", NOSSO[ds]):
        lista = json.load(open(out / f"pruning_{nome}" / "selected.json"))["features"]
        quadros = {fam: pd.read_parquet(feats / f"{fam}.parquet", columns=["ecg_id", *cs]).set_index("ecg_id")
                   for fam, cs in partes(lista).items()}
        yield (nome, montar(quadros, lista, ids), y, rot,
               json.load(open(out / f"boosting_{nome}_pruned.json"))["test_macro_auroc"], out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+", choices=["ptbxl", "chapman8", "code15"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--sets", nargs="+", default=None,
                    help="so estes conjuntos (ex.: todas_familias); padrao: catch22, tsfresh e o nosso")
    args = ap.parse_args()
    resumo_arq = ROOT / "results/boosting_seeds_fold10.json"
    resumo = json.load(open(resumo_arq)) if resumo_arq.exists() else {}
    for ds in args.datasets:
        for nome, x, y, rot, gravada, out in problemas(ds):
            if args.sets and nome not in args.sets:
                continue
            chave = f"{ds}/{'nosso' if nome == NOSSO.get(ds) else nome}"
            (out / "seeds").mkdir(exist_ok=True)
            aucs = []
            for s in SEMENTES:
                with threadpool_limits(args.threads):
                    m = BoostingPorRotulo(semente=s, max_iter=2000, **BOOSTING).fit(
                        x["train"], y["train"], x["validation"], y["validation"])
                    esc = m.predict_proba(x["test"])
                auc = float(np.mean([roc_auc_score(y["test"][:, i], esc[:, i]) for i in range(len(rot))]))
                if s == 42 and abs(auc - gravada) > 1e-9:
                    print(f"AVISO {chave}: semente 42 deu {auc:.6f}, gravada {gravada:.6f}")
                aucs.append(auc)
                np.savez(out / "seeds" / f"boosting_{nome}_seed{s}_scores.npz", test=esc, y_test=y["test"])
                with (ROOT / "results/test_fold_usage.jsonl").open("a") as f:
                    f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                        "reason": MOTIVO.format(s=s), "test_fold": 10,
                                        "model": f"boosting {nome} semente {s} [{ds}]", "dataset": ds,
                                        "feature_count": int(x["test"].shape[1]), "test_macro_auroc": auc,
                                        "output_dir": str(out / "seeds")}) + "\n")
                print(f"{chave:22s} semente {s}  teste {auc:.4f}", flush=True)
            resumo[chave] = {"conjunto": nome, "k": int(x["test"].shape[1]), "por_semente": aucs,
                             "media": float(np.mean(aucs)), "dp": float(np.std(aucs, ddof=1)),
                             "semente42_gravada": gravada}
            resumo_arq.write_text(json.dumps(resumo, indent=2) + "\n")
            print(f"== {chave}: {np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}", flush=True)


if __name__ == "__main__":
    main()
