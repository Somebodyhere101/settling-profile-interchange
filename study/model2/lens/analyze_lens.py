"""Paper 8 lens-robustness arm: analysis and decision rule (lens/PREDICTIONS.md §4).

python analyze_lens.py [smoke]  ->  res/lens_summary.json

Per item and prompt form, per digit: settle layer under (a) the logit lens and (b) the tuned lens; (c) causal
arrival = smallest l such that, for every l' >= l, patching the clean block-l' output at the digit's predicting
position into the corrupt partner's run makes the clean digit top-1 (stable arrival). Also: first arrival, and the
stable 50% logit-difference layer (normalised (LD - LD_corrupt) / (LD_clean - LD_corrupt) >= 0.5).
Item measure: mean over the item's digits (in layers), as D. Primary: original form, all items.
CIs: item bootstrap, 10,000 resamples, seed 90 (Spearman); S2 grouping: two-way item bootstrap, seed 80.
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
L = 24
NB = 10000


def stable_first(flags):
    l = len(flags)
    while l > 0 and flags[l - 1]:
        l -= 1
    return l if l < len(flags) else len(flags)  # len(flags) = never (cannot happen: last block reproduces the output)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))]


recs = [json.loads(l) for l in open(os.path.join(RES, "lens" + ("_smoke" if SMOKE else "") + ".jsonl"), encoding="utf-8")]
M = {}
for r in recs:
    nd = len(r["ll"])
    arr = [stable_first([bool(t) for t in r["top"][k]]) for k in range(nd)]
    first = [next((l for l, t in enumerate(r["top"][k]) if t), L) for k in range(nd)]
    a50 = []
    for k in range(nd):
        den = r["ld_clean"][k] - r["ld_corrupt"][k]
        fl = [(x - r["ld_corrupt"][k]) / den >= 0.5 if den > 0 else False for x in r["ld"][k]]
        a50.append(stable_first(fl))
    M[(r["id"], r["form"])] = dict(ll=float(np.mean(r["ll"])), tl=float(np.mean(r["tl"])), c=float(np.mean(arr)),
                                   c_first=float(np.mean(first)), c50=float(np.mean(a50)),
                                   ll_gen=float(np.mean(r["ll"][1:])), tl_gen=float(np.mean(r["tl"][1:])), c_gen=float(np.mean(arr[1:])),
                                   digits=dict(ll=r["ll"], tl=r["tl"], c=arr), s2=r["s2"], set=r["set"], ok=r["argmax_ok"],
                                   corrupt_ok=r["corrupt_argmax_is_own"])
ids = sorted({k[0] for k in M if k[1] == "orig"})
rng = np.random.default_rng(90)
BI = [rng.integers(0, len(ids), len(ids)) for _ in range(NB)]


def rho(xa, xb, sub=None, form="orig", boot=True):
    ii = [i for i in ids if (sub is None or sub(M[(i, form)])) and (i, form) in M]
    a = np.array([M[(i, form)][xa] for i in ii])
    b = np.array([M[(i, form)][xb] for i in ii])
    est = float(spearmanr(a, b)[0])
    if not boot:
        return dict(rho=est, n=len(ii))
    r2 = np.random.default_rng(90)
    bs = [spearmanr(a[j], b[j])[0] for j in (r2.integers(0, len(ii), len(ii)) for _ in range(NB))]
    return dict(rho=est, ci=ci(bs), n=len(ii))


out = dict(n_items=len(ids), n_rows=len(recs))
out["primary"] = {"a_c": rho("ll", "c"), "b_c": rho("tl", "c"), "a_b": rho("ll", "tl")}
out["secondary_measures"] = {
    "a_c50": rho("ll", "c50"), "b_c50": rho("tl", "c50"), "a_cfirst": rho("ll", "c_first"), "b_cfirst": rho("tl", "c_first"),
    "a_c_gen": rho("ll_gen", "c_gen"), "b_c_gen": rho("tl_gen", "c_gen"),
    "a_c_S2only": rho("ll", "c", sub=lambda x: x["s2"] == 1), "a_c_fresh": rho("ll", "c", sub=lambda x: x["s2"] == 0),
    "b_c_S2only": rho("tl", "c", sub=lambda x: x["s2"] == 1), "b_c_fresh": rho("tl", "c", sub=lambda x: x["s2"] == 0),
    "a_c_freshM": rho("ll", "c", sub=lambda x: x["s2"] == 0 and x["set"] == "M"),
}
dl = [(d_ll, d_c) for i in ids for d_ll, d_c in zip(M[(i, "orig")]["digits"]["ll"], M[(i, "orig")]["digits"]["c"])]
dt = [(d_tl, d_c) for i in ids for d_tl, d_c in zip(M[(i, "orig")]["digits"]["tl"], M[(i, "orig")]["digits"]["c"])]
out["digit_level"] = dict(a_c=float(spearmanr(*zip(*dl))[0]), b_c=float(spearmanr(*zip(*dt))[0]), n_digits=len(dl))
out["means_layers"] = {k: float(np.mean([M[(i, "orig")][k] for i in ids])) for k in ("ll", "tl", "c", "c50", "c_first")}
out["sd_layers"] = {k: float(np.std([M[(i, "orig")][k] for i in ids])) for k in ("ll", "tl", "c", "c50", "c_first")}
out["checks"] = dict(clean_argmax_ok=float(np.mean([M[(i, "orig")]["ok"] for i in ids])),
                     corrupt_argmax_is_own=float(np.mean([M[(i, "orig")]["corrupt_ok"] for i in ids])))
# reliability across the two prompt forms
both = [i for i in ids if (i, "para") in M]
out["reliability_orig_vs_para"] = {k: dict(rho=float(spearmanr([M[(i, "orig")][k] for i in both], [M[(i, "para")][k] for i in both])[0]), n=len(both))
                                   for k in ("ll", "tl", "c", "c50")} if both else None

# S2 grouping: does the far/null split still separate items under each measure? (s-signed far - null, donor - recipient)
pairs = json.load(open(os.path.join(HERE, "..", "patch2", "res", "pairs2.json"), encoding="utf-8"))["S"]
pairs = [p for p in pairs if (p["r"], "orig") in M and all((d, "orig") in M for d in p["far"] + p["null"])]
pid = sorted({p["r"] for p in pairs} | {d for p in pairs for d in p["far"] + p["null"]})
IX = {i: k for k, i in enumerate(pid)}
Wi = np.random.default_rng(80).poisson(1.0, (NB, len(pid))).astype(float)


def s2_gap(key):
    num, den, pts = np.zeros(NB), np.zeros(NB), []
    for p in pairs:
        f = [(IX[d], M[(d, "orig")][key] - M[(p["r"], "orig")][key]) for d in p["far"]]
        n = [(IX[d], M[(d, "orig")][key] - M[(p["r"], "orig")][key]) for d in p["null"]]
        pts.append(p["s"] * (np.mean([x for _, x in f]) - np.mean([x for _, x in n])))
        wf = np.stack([Wi[:, i] for i, _ in f], 1)
        wn = np.stack([Wi[:, i] for i, _ in n], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            e = p["s"] * ((wf * [x for _, x in f]).sum(1) / wf.sum(1) - (wn * [x for _, x in n]).sum(1) / wn.sum(1))
        w = Wi[:, IX[p["r"]]] * np.isfinite(e)
        num += w * np.nan_to_num(e)
        den += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(gap_layers=float(np.mean(pts)), ci=ci(num / den), n=len(pts))


out["s2_grouping"] = {k: s2_gap(k) for k in ("ll", "tl", "c", "c50", "ll_gen", "tl_gen", "c_gen")} if pairs else None

# decision rule (§4)
ac, bc = out["primary"]["a_c"], out["primary"]["b_c"]
if ac["rho"] >= 0.5 and ac["ci"][0] > 0.3:
    verdict = "ROBUST PROCESS VARIABLE"
elif ac["rho"] < 0.2 and bc["rho"] > ac["rho"]:
    verdict = "LENS ARTEFACT"
else:
    verdict = "MIXED"
out["verdict"] = verdict
json.dump(out, open(os.path.join(RES, "lens_summary" + ("_smoke" if SMOKE else "") + ".json"), "w"), indent=1)
print(json.dumps(out, indent=1))
