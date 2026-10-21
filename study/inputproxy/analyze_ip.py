"""Paper 8 input-proxy arm: analysis and decision rule (inputproxy/PREDICTIONS.md §4).

python analyze_ip.py  ->  res/ip_summary.json

T(i, v): the mean over the 4 digits of the causal arrival layer with corrupt run A (primary target); T_B the same
with corrupt run B (reliability). Items enter if v0 and at least 2 other variants were kept.
Between items (v0 only): oracle o = mean feature of the top tercile of T minus the bottom tercile, fitted on 4/5
of the items and applied to the held-out fifth (5 folds, seed 12); between-item AUC (top vs bottom tercile,
thresholds from the training folds) and Spearman on the held-out projections, pooled over folds.
Within items: each item's projections (its fold's o) and T, centred over its kept variants; rho_w = Spearman of
the pooled centred values; CI: item bootstrap, 10,000 resamples, seed 13.
Reliability: Pearson of within-item-centred T_A against T_B.
"""

import json
import os

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
NB = 10000
SITES = ["r21", "r14", "e21", "e14"]
PRIMARY = "r21"

recs = [json.loads(l) for l in open(os.path.join(RES, "ip.jsonl"), encoding="utf-8")]
H = dict(np.load(os.path.join(RES, "ip_h.npz")))
by = {}
for r in recs:
    by.setdefault(r["id"], {})[r["v"]] = r
items = sorted(i for i, v in by.items() if "v0" in v and len(v) >= 3)
T = {(i, v): float(np.mean(by[i][v]["arrA"])) for i in items for v in by[i]}
TB = {(i, v): float(np.mean(by[i][v]["arrB"])) for i in items for v in by[i]}
TS = {(i, v): float(np.mean(by[i][v]["settle"])) for i in items for v in by[i]}


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


def analyse(site, target):
    rng = np.random.default_rng(12)
    fold = {i: f for i, f in zip(items, rng.integers(0, 5, len(items)))}
    proj, auc_s, auc_y, sp_x, sp_y = {}, [], [], [], []
    for f in range(5):
        tr = [i for i in items if fold[i] != f]
        te = [i for i in items if fold[i] == f]
        Xtr = np.array([H[f"{i}|v0|{site}"] for i in tr], float)
        ytr = np.array([target[(i, "v0")] for i in tr])
        lo, hi = np.quantile(ytr, [1 / 3, 2 / 3])
        o = Xtr[ytr >= hi].mean(0) - Xtr[ytr <= lo].mean(0)
        o /= np.linalg.norm(o)
        for i in te:
            for v in by[i]:
                proj[(i, v)] = float(H[f"{i}|{v}|{site}"] @ o)
            t0 = target[(i, "v0")]
            sp_x.append(proj[(i, "v0")])
            sp_y.append(t0)
            if t0 >= hi or t0 <= lo:
                auc_s.append(proj[(i, "v0")])
                auc_y.append(int(t0 >= hi))
    between_auc = auc(auc_s, auc_y)
    between_rho = float(spearmanr(sp_x, sp_y)[0])
    # within items
    cx, cy, owner = [], [], []
    for i in items:
        vs = sorted(by[i])
        px = np.array([proj[(i, v)] for v in vs])
        ty = np.array([target[(i, v)] for v in vs])
        cx += list(px - px.mean())
        cy += list(ty - ty.mean())
        owner += [i] * len(vs)
    cx, cy, owner = np.array(cx), np.array(cy), np.array(owner)
    rho_w = float(spearmanr(cx, cy)[0])
    groups = {i: np.where(owner == i)[0] for i in items}
    rb = np.random.default_rng(13)
    bs = []
    for _ in range(NB):
        pick = rb.integers(0, len(items), len(items))
        idx = np.concatenate([groups[items[k]] for k in pick])
        bs.append(spearmanr(cx[idx], cy[idx])[0])
    return dict(between_auc=between_auc, between_rho=between_rho, n_auc=len(auc_y), rho_w=rho_w, rho_w_ci=ci(bs), n_within=len(cx))


out = dict(n_items=len(items), n_rows=sum(len(by[i]) for i in items),
           variants_kept={v: sum(v in by[i] for i in items) for v in ["v0", "v1", "v2", "v3", "v4", "v5", "v6"]})
# target reliability and variance decomposition
ca, cb, wa = [], [], []
for i in items:
    vs = sorted(by[i])
    a = np.array([T[(i, v)] for v in vs])
    b = np.array([TB[(i, v)] for v in vs])
    ca += list(a - a.mean())
    cb += list(b - b.mean())
out["reliability_within"] = dict(pearson=float(pearsonr(ca, cb)[0]), spearman=float(spearmanr(ca, cb)[0]))
out["reliability_between_v0"] = float(pearsonr([T[(i, "v0")] for i in items], [TB[(i, "v0")] for i in items])[0])
allT = np.array([T[(i, v)] for i in items for v in by[i]])
within_var = float(np.mean([np.var([T[(i, v)] for v in by[i]]) for i in items]))
out["variance"] = dict(total=float(allT.var()), within_item_mean=within_var, between_item_v0=float(np.var([T[(i, "v0")] for i in items])),
                       mean_T=float(allT.mean()))
out["variant_shift_vs_v0"] = {}
for v in ["v1", "v2", "v3", "v4", "v5", "v6"]:
    dd = np.array([T[(i, v)] - T[(i, "v0")] for i in items if v in by[i]])
    if len(dd):
        rb = np.random.default_rng(14)
        out["variant_shift_vs_v0"][v] = dict(mean=float(dd.mean()), ci=ci([dd[rb.integers(0, len(dd), len(dd))].mean() for _ in range(2000)]), n=len(dd))
out["sites"] = {s: analyse(s, T) for s in SITES}
out["sites_settle_target"] = {s: analyse(s, TS) for s in SITES}
out["settle_vs_arrival_rho"] = float(spearmanr([TS[k] for k in T], [T[k] for k in T])[0])

p = out["sites"][PRIMARY]
if out["reliability_within"]["pearson"] < 0.5:
    verdict = "UNDERPOWERED (within-item differences of the target are not reliable: split-half r < 0.5)"
elif p["rho_w_ci"][1] < 0.15 and p["between_auc"] >= 0.75:
    verdict = "INPUT-PROXY"
elif p["rho_w"] >= 0.3 and p["rho_w_ci"][0] > 0:
    verdict = "PROCESS-READABLE"
elif p["between_auc"] < 0.75:
    verdict = "NOT READABLE BETWEEN ITEMS (between-item AUC < 0.75)"
else:
    verdict = "INCONCLUSIVE"
out["verdict"] = verdict
json.dump(out, open(os.path.join(RES, "ip_summary.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
