"""Paper 8 self-report arm: analysis and decision rules (selfreport/PREDICTIONS.md §4-§5).

python analyze_sr.py kill [smoke]   ->  res/kill_summary.json (verdict PASS / KILL)
python analyze_sr.py sr             ->  res/sr_summary.json

Contrast of a cell on a per-row value X (as ../patch2/analyze_patch2.py): for each recipient,
s_r * (mean X over far-donor rows - mean X over null-donor rows), averaged over recipients; s_r = +1 for an
early recipient, -1 for a late one. Gap: the same contrast on the clean donor rows. tau = effect / gap.
CIs: two-way item bootstrap (Poisson(1) weight per item, used when it is a recipient and when it is a
donor), 10,000 resamples, seed 80, percentile 95%; ratios use the same resamples.
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
WHAT = sys.argv[1]
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
L = 28
NB = 10000
KILL_TAU = 0.2


def m_Dgen(st, l=None):
    return np.mean(st[1:]) / L if len(st) > 1 else np.nan


def m_D(st, l=None):
    return np.mean(st) / L if st else np.nan


def m_Dgen_above(st, l):
    return np.mean([max(s, l + 1) for s in st[1:]]) / L if len(st) > 1 else np.nan


def ci(x):
    x = np.asarray(x)[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


class Boot:
    """Two-way item bootstrap over recipients; values are per recipient: far list, null list."""

    def __init__(self, recs):
        rng = np.random.default_rng(80)
        self.recs = recs
        ids = sorted({r["r"] for r in recs} | {dd for r in recs for dd in r["dons"]})
        self.IX = {i: k for k, i in enumerate(ids)}
        self.Wi = rng.poisson(1.0, (NB, len(ids))).astype(float)
        self.S = np.array([r["s"] for r in recs], float)
        self.IR = np.array([self.IX[r["r"]] for r in recs])

    def contrast(self, val):
        """val(rec, donor) -> value or nan. Returns point estimate and bootstrap distribution."""
        R = len(self.recs)
        out = {}
        for role in ("far", "null"):
            lists = [[dd for dd in r["dons"] if r["dons"][dd] == role] for r in self.recs]
            K = max(len(x) for x in lists)
            V = np.full((R, K), np.nan)
            I = np.zeros((R, K), int)
            for a, (r, ds) in enumerate(zip(self.recs, lists)):
                for k, dd in enumerate(ds):
                    v = val(r, dd)
                    V[a, k] = np.nan if v is None else v
                    I[a, k] = self.IX[dd]
            out[role] = (V, I)
        (F, iF), (N, iN) = out["far"], out["null"]
        with np.errstate(invalid="ignore"):
            e = self.S * (np.nanmean(F, 1) - np.nanmean(N, 1))
        ok = np.isfinite(e)
        point = float(np.mean(e[ok]))

        def wmean(V, W):
            okk = ~np.isnan(V)
            W = W * okk
            sw = W.sum(-1)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(sw > 0, (W * np.nan_to_num(V)).sum(-1) / sw, np.nan)
        f = wmean(F[None], self.Wi[:, iF])
        n = wmean(N[None], self.Wi[:, iN])
        eb = self.S[None] * (f - n)
        w = self.Wi[:, self.IR] * np.isfinite(eb)
        with np.errstate(invalid="ignore", divide="ignore"):
            two = (w * np.nan_to_num(eb)).sum(1) / w.sum(1)
        return dict(point=point, boot=two, n=int(ok.sum()))


def summ(C, G=None):
    d = dict(effect=C["point"], ci=ci(C["boot"]), n=C["n"])
    if G is not None:
        d["gap"] = G["point"]
        d["gap_ci"] = ci(G["boot"])
        d["tau"] = C["point"] / G["point"] if G["point"] else None
        with np.errstate(invalid="ignore", divide="ignore"):
            d["tau_ci"] = ci(C["boot"] / G["boot"])
    return d


# ======================================================================= kill
if WHAT == "kill":
    recs = [json.loads(l) for l in open(os.path.join(RES, "kill" + ("_smoke" if SMOKE else "") + ".jsonl"), encoding="utf-8")]
    for r in recs:
        r["dons"] = {dd: v["role"] for dd, v in r["don"].items()}
        r["cell"] = {(x["cond"], x["l"], x["donor"]): x for x in r["rows"]}
    bt = Boot(recs)
    out = dict(n_recipients=len(recs), cells={})
    for lens in ("ll", "tl"):
        for mname, mf in (("D_gen", m_Dgen), ("D", m_D), ("D_gen_above", m_Dgen_above)):
            for cond, l in (("iv", 21), ("iv", 14), ("iiip", 21)):
                G = bt.contrast(lambda r, dd: mf(r["don"][dd][lens], l) - mf(r["clean"][lens], l))
                C = bt.contrast(lambda r, dd: mf(r["cell"][(cond, l, dd)][lens], l) - mf(r["clean"][lens], l))
                out["cells"][f"{lens}|{mname}|{cond}@{l}"] = summ(C, G)
    out["same_answer_iv21"] = float(np.mean([x["same"] for r in recs for x in r["rows"] if x["cond"] == "iv" and x["l"] == 21]))
    out["clean_settle_mean"] = dict(ll=float(np.mean([s for r in recs for s in r["clean"]["ll"]])),
                                    tl=float(np.mean([s for r in recs for s in r["clean"]["tl"]])))
    rho = spearmanr([m_D(r["clean"]["ll"]) for r in recs], [m_D(r["clean"]["tl"]) for r in recs])[0]
    out["rho_D_logit_vs_tuned_recipients"] = float(rho)
    k = out["cells"]["tl|D_gen|iv@21"]
    out["verdict"] = "PASS" if (k["tau"] is not None and k["tau"] >= KILL_TAU) else "KILL"
    json.dump(out, open(os.path.join(RES, "kill_summary" + ("_smoke" if SMOKE else "") + ".json"), "w"), indent=1)
    for key, v in out["cells"].items():
        print(f"{key:24s} effect {v['effect']:+.4f} {['%+.4f' % x for x in v['ci']]} gap {v['gap']:+.4f} tau {v['tau']:.2f} {['%.2f' % x for x in v['tau_ci']]}")
    print({k: v for k, v in out.items() if k != "cells"})
    sys.exit(0)

# ========================================================================= sr
assert WHAT == "sr"
raw = [json.loads(l) for l in open(os.path.join(RES, "sr.jsonl"), encoding="utf-8")]
by = {}
for x in raw:
    by.setdefault(x["r"], {})[x["order"]] = x
recs = []
for rid, od in by.items():
    if set(od) != {0, 1}:
        continue
    a, b = od[0], od[1]
    rec = dict(r=rid, s=a["s"], dons={dd: v["role"] for dd, v in a["don"].items()},
               clean=dict(r=(a["clean"]["r"] + b["clean"]["r"]) / 2, ll=a["clean"]["ll"], tl=a["clean"]["tl"]),
               don={dd: dict(r=(a["don"][dd]["r"] + b["don"][dd]["r"]) / 2, ll=a["don"][dd]["ll"], tl=a["don"][dd]["tl"]) for dd in a["don"]},
               cell={}, gauss={0: [], 1: []}, raw={0: a, 1: b})
    for o, x in ((0, a), (1, b)):
        for rw in x["rows"]:
            if "gj" in rw:
                rec["gauss"][o].append(rw)
            else:
                key = (rw["cond"], rw["l"], rw["donor"])
                cur = rec["cell"].setdefault(key, dict(r=0.0, kl=0.0, ll=rw["ll"], tl=rw["tl"]))
                cur["r"] += rw["r"] / 2
                cur["kl"] += rw["kl"] / 2
                cur[f"r{o}"], cur[f"kl{o}"] = rw["r"], rw["kl"]
    recs.append(rec)
bt = Boot(recs)
out = dict(n_recipients=len(recs))

# clean: Spearman(r, D) on recipients (r averaged over the two label orders)
rr = np.array([r["clean"]["r"] for r in recs])
DD = np.array([m_D(r["clean"]["ll"]) for r in recs])
rng = np.random.default_rng(81)
bs = [spearmanr(rr[i], DD[i])[0] for i in (rng.integers(0, len(rr), len(rr)) for _ in range(NB))]
out["clean_spearman_r_D"] = dict(rho=float(spearmanr(rr, DD)[0]), ci=ci(bs), n=len(rr),
                                 rho_Dgen=float(spearmanr(rr, [m_Dgen(r["clean"]["ll"]) for r in recs])[0]),
                                 rho_D_tuned=float(spearmanr(rr, [m_D(r["clean"]["tl"]) for r in recs])[0]))
items_r, items_D = {}, {}
for r in recs:
    items_r.setdefault(r["r"], []).append(r["clean"]["r"])
    items_D[r["r"]] = m_D(r["clean"]["ll"])
    for dd, v in r["don"].items():
        items_r.setdefault(dd, []).append(v["r"])
        items_D[dd] = m_D(v["ll"])
ks_ = sorted(items_r)
out["clean_spearman_all_items"] = dict(rho=float(spearmanr([np.mean(items_r[k]) for k in ks_], [items_D[k] for k in ks_])[0]), n=len(ks_))
out["clean_r"] = dict(mean=float(rr.mean()), sd=float(rr.std()),
                      order0=float(np.mean([r["raw"][0]["clean"]["r"] for r in recs])),
                      order1=float(np.mean([r["raw"][1]["clean"]["r"] for r in recs])),
                      pab_min=float(min(min(r["raw"][o]["clean"]["pab"] for o in (0, 1)) for r in recs)))

Gr = bt.contrast(lambda r, dd: r["don"][dd]["r"] - r["clean"]["r"])
GD = bt.contrast(lambda r, dd: m_Dgen(r["don"][dd]["ll"]) - m_Dgen(r["clean"]["ll"]))
out["gap_r"] = summ(Gr)
out["cells"] = {}
for cond, l in (("iv", 21), ("i", 7), ("iir", 21), ("iir", 7), ("ii", 7)):
    Cr = bt.contrast(lambda r, dd: r["cell"][(cond, l, dd)]["r"] - r["clean"]["r"])
    CD = bt.contrast(lambda r, dd: m_Dgen(r["cell"][(cond, l, dd)]["ll"]) - m_Dgen(r["clean"]["ll"]))
    out["cells"][f"{cond}@{l}"] = dict(r=summ(Cr, Gr), D_gen=summ(CD, GD),
                                       kl_median=float(np.median([v["kl"] for r in recs for k, v in r["cell"].items() if k[:2] == (cond, l)])))


# matched-KL Gaussian control at the (iv)@21 positions: per recipient and order, the r shift of each
# Gaussian direction interpolated in log KL at the row's KL; averaged over directions, then over orders
def g_at(rec, o, kl):
    vals = []
    for j in sorted({g["gj"] for g in rec["gauss"][o]}):
        pts = sorted((max(g["kl"], 1e-12), g["r"] - rec["raw"][o]["clean"]["r"]) for g in rec["gauss"][o] if g["gj"] == j)
        xs = [math.log(p[0]) for p in pts]
        lk = math.log(max(kl, 1e-12))
        if lk < xs[0] or lk > xs[-1]:
            continue
        for (x0, (_, v0)), (x1, (_, v1)) in zip(zip(xs, pts), zip(xs[1:], pts[1:])):
            if x0 <= lk <= x1:
                vals.append(v0 if x1 == x0 else v0 + (v1 - v0) * (lk - x0) / (x1 - x0))
                break
    return float(np.mean(vals)) if vals else None


def corrected(r, dd):
    cell = r["cell"][("iv", 21, dd)]
    vs = []
    for o in (0, 1):
        g = g_at(r, o, cell[f"kl{o}"])
        if g is None:
            return None
        vs.append(cell[f"r{o}"] - r["raw"][o]["clean"]["r"] - g)
    return float(np.mean(vs))


Cc = bt.contrast(corrected)
n_excl = sum(corrected(r, dd) is None for r in recs for dd in r["dons"])
out["iv21_kl_corrected"] = dict(**summ(Cc, Gr), rows_not_bracketed=n_excl)
out["gauss_r_shift_by_c"] = {str(c): float(np.mean([g["r"] - r["raw"][o]["clean"]["r"] for r in recs for o in (0, 1) for g in r["gauss"][o] if g["gc"] == c]))
                             for c in sorted({g["gc"] for r in recs for o in (0, 1) for g in r["gauss"][o]})}
out["gauss_kl_by_c"] = {str(c): float(np.median([g["kl"] for r in recs for o in (0, 1) for g in r["gauss"][o] if g["gc"] == c]))
                        for c in sorted({g["gc"] for r in recs for o in (0, 1) for g in r["gauss"][o]})}

# ------------------------------------------------------------- decision rules (§5)
c4, c1, c2 = out["cells"]["iv@21"], out["cells"]["i@7"], out["cells"]["ii@7"]
tauD = c4["D_gen"]["tau"]
gap_ok = out["gap_r"]["ci"][0] is not None and (out["gap_r"]["ci"][0] > 0 or out["gap_r"]["ci"][1] < 0)


def moves(c):
    e, (lo, hi) = c["r"]["effect"], c["r"]["ci"]
    return (lo > 0 or hi < 0) and c["r"]["tau"] is not None and abs(c["r"]["tau"]) >= 0.2


def toward(c):  # CI excludes 0 in the direction of the gap
    lo, hi = c["ci"]
    return (lo > 0) if out["gap_r"]["effect"] > 0 else (hi < 0)


faithful = (gap_ok and c4["r"]["tau"] is not None and c4["r"]["tau"] >= 0.5 * tauD and toward(c4["r"])
            and toward(out["iv21_kl_corrected"]) and not moves(c1))
reconstruction = gap_ok and (moves(c1) or moves(c2)) and not (c4["r"]["tau"] is not None and c4["r"]["tau"] >= 0.5 * tauD and toward(c4["r"]))
if not gap_ok:
    verdict = "NOT EVALUABLE (the self-report does not differ between far and null donors: gap in r CI includes 0)"
elif faithful:
    verdict = "FAITHFUL PROCESS ACCESS"
elif reconstruction:
    verdict = "RECONSTRUCTION FROM INPUT"
else:
    verdict = "INCONCLUSIVE"
out["tau_D_iv21"] = tauD
out["verdict"] = verdict
out["inert_check_iir"] = {k: out["cells"][k]["r"]["ci"] for k in ("iir@21", "iir@7")}
json.dump(out, open(os.path.join(RES, "sr_summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "cells"}, indent=1, default=float))
for k, v in out["cells"].items():
    print(k, "r:", {x: v["r"][x] for x in ("effect", "ci", "tau", "tau_ci")}, "| D_gen tau", round(v["D_gen"]["tau"], 3), "| kl~", round(v["kl_median"], 2))
