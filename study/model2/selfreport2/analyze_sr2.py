"""Paper 8 self-report arm 2: analysis and decision rules (selfreport2/PREDICTIONS.md §4-§5).

python analyze_sr2.py  ->  res/sr2_summary.json

Primary (digit question). For a row, the report distribution p over digits 1-4 is the softmax of the
digit logits averaged over the 4 option rotations (each rotation's 4 letter logits centred first).
For a (recipient R, donor D) pair, w = settle_D - settle_R per digit (logit-lens settle layers of the clean
rows of the same pass), with digit 1 set to 0 ((iv) never touches digit 1). y(p) = sum_k p_k w_k (layers).
Row effect: y(p_row) - y(p_R). Pair gap: y(p_D) - y(p_R). Pairs with w == 0 are excluded (counted).
Cell effect: mean over recipients of the mean over its donors; tau = effect / gap; CIs: two-way item
bootstrap (Poisson(1) weight per item, as recipient and as donor), 10,000 resamples, seed 80.
"""

import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SR2_RES", os.path.join(HERE, "res"))
L = 24
NB = 10000
CELLS = ["iv@18", "ivrep@18", "i@6", "iir@18", "ii@6"]

raw = [json.loads(l) for l in open(os.path.join(RES, "sr2.jsonl"), encoding="utf-8")]
by = {}
for x in raw:
    by.setdefault(x["r"], {})[(x["kind"], x["k"])] = x
recs = []
for rid, t in by.items():
    if len(t) != 6:
        continue
    t0 = t[("digit", 0)]
    rec = dict(r=rid, s=t0["s"], dons={dd: v["role"] for dd, v in t0["don"].items()},
               sR=t0["clean"]["ll"], sD={dd: v["ll"] for dd, v in t0["don"].items()}, t=t)
    recs.append(rec)
R = len(recs)
rng = np.random.default_rng(80)
ids = sorted({r["r"] for r in recs} | {dd for r in recs for dd in r["dons"]})
IX = {i: k for k, i in enumerate(ids)}
Wi = rng.poisson(1.0, (NB, len(ids))).astype(float)


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))] if len(x) else [None, None]


def softmax(u):
    e = np.exp(u - np.max(u))
    return e / e.sum()


def digit_u(entries):
    """entries: {k: 4 letter logits} for rotations k -> centred digit logits averaged over rotations."""
    acc = np.zeros(4)
    for k, lg in entries.items():
        lg = np.asarray(lg) - np.mean(lg)
        acc += np.array([lg[(dg - k) % 4] for dg in range(4)])
    return acc / len(entries)


def row_key(rw):
    return (rw["cond"], rw["l"], rw["donor"]) if "cond" in rw else ("g", rw["gj"], rw["gc"])


for rec in recs:
    U, KL, LL, SAME = {}, {}, {}, {}
    rows_by_k = {k: {row_key(rw): rw for rw in rec["t"][("digit", k)]["rows"]} for k in range(4)}
    for key in rows_by_k[0]:
        U[key] = digit_u({k: rows_by_k[k][key]["lg"] for k in range(4)})
        KL[key] = float(np.mean([rows_by_k[k][key]["kl"] for k in range(4)]))
        LL[key] = rows_by_k[0][key]["ll"]
        SAME[key] = rows_by_k[0][key]["same"]
    rec["U"], rec["KL"], rec["LL"], rec["SAME"] = U, KL, LL, SAME
    rec["uR"] = digit_u({k: rec["t"][("digit", k)]["clean"]["lg"] for k in range(4)})
    rec["uD"] = {dd: digit_u({k: rec["t"][("digit", k)]["don"][dd]["lg"] for k in range(4)}) for dd in rec["dons"]}
    rec["mass_min"] = min(min(rec["t"][("digit", k)]["clean"]["mass"] for k in range(4)), 1.0)
    # A/B (secondary): r = step-by-step minus just-knew, mean of the two orders
    ab = {}
    for o in (0, 1):
        sg = 1.0 if o == 0 else -1.0
        x = rec["t"][("ab", o)]
        ab.setdefault("R", []).append(sg * (x["clean"]["lg"][0] - x["clean"]["lg"][1]))
        for dd, v in x["don"].items():
            ab.setdefault(("D", dd), []).append(sg * (v["lg"][0] - v["lg"][1]))
        for rw in x["rows"]:
            ab.setdefault(row_key(rw), []).append(sg * (rw["lg"][0] - rw["lg"][1]))
    rec["ab"] = {k: float(np.mean(v)) for k, v in ab.items()}


def w_of(rec, dd):
    w = np.asarray(rec["sD"][dd], float) - np.asarray(rec["sR"], float)
    w[0] = 0.0
    return w


def y(u, w):
    return float(softmax(u) @ w)


def pairwise(val):
    """val(rec, donor) -> value or None. Mean over donors, then over recipients; two-way bootstrap."""
    V = [[(IX[dd], val(r, dd)) for dd in r["dons"]] for r in recs]
    rec_mean, okr = [], []
    for v in V:
        vv = [x for _, x in v if x is not None and np.isfinite(x)]
        rec_mean.append(np.mean(vv) if vv else np.nan)
    rec_mean = np.array(rec_mean)
    ok = np.isfinite(rec_mean)
    point = float(np.mean(rec_mean[ok]))
    boot = np.zeros(NB)
    num = np.zeros(NB)
    den = np.zeros(NB)
    for a, v in enumerate(V):
        vv = [(i, x) for i, x in v if x is not None and np.isfinite(x)]
        if not vv:
            continue
        wd = np.stack([Wi[:, i] for i, _ in vv], 1)
        xs = np.array([x for _, x in vv])
        sw = wd.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rm = (wd * xs).sum(1) / sw
        wr = Wi[:, IX[recs[a]["r"]]] * (sw > 0)
        num += wr * np.nan_to_num(rm)
        den += wr
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num / den
    return dict(point=point, boot=boot, n_rec=int(ok.sum()), n_pairs=int(sum(1 for v in V for _, x in v if x is not None and np.isfinite(x))))


def summ(C, G=None):
    d = dict(effect=C["point"], ci=ci(C["boot"]), n_rec=C["n_rec"], n_pairs=C["n_pairs"])
    if G is not None:
        d.update(gap=G["point"], gap_ci=ci(G["boot"]), tau=C["point"] / G["point"] if G["point"] else None)
        with np.errstate(invalid="ignore", divide="ignore"):
            d["tau_ci"] = ci(C["boot"] / G["boot"])
    return d


def informative(rec, dd):
    return bool(np.any(w_of(rec, dd) != 0))


out = dict(n_recipients=R, n_items=len(ids))

# ------------------------------------------------------------- clean: does the report follow settling?
self_y, top_hit, top_chance = [], [], []
items_seen = set()
for rec in recs:
    for name, u, s in [(rec["r"], rec["uR"], rec["sR"])] + [(dd, rec["uD"][dd], rec["sD"][dd]) for dd in rec["dons"]]:
        if name in items_seen:
            continue
        items_seen.add(name)
        s = np.asarray(s, float)
        self_y.append(float(softmax(u) @ (s - s.mean())))
        latest = set(np.flatnonzero(s == s.max()).tolist())
        top_hit.append(int(int(np.argmax(u)) in latest))
        top_chance.append(len(latest) / 4)
rb = np.random.default_rng(81)
self_y = np.array(self_y)
out["clean_report_vs_own_settle"] = dict(
    mean_y_self_layers=float(self_y.mean()), ci=ci([self_y[rb.integers(0, len(self_y), len(self_y))].mean() for _ in range(NB)]),
    argmax_is_latest_rate=float(np.mean(top_hit)), chance=float(np.mean(top_chance)), n_items=len(self_y),
    report_argmax_counts={str(k): int(v) for k, v in zip(*np.unique([int(np.argmax(r["uR"])) + 1 for r in recs], return_counts=True))})
out["mass_min_clean_digit"] = float(min(r["mass_min"] for r in recs))
out["pairs_informative"] = int(sum(informative(r, dd) for r in recs for dd in r["dons"]))
out["pairs_total"] = int(sum(len(r["dons"]) for r in recs))

G = pairwise(lambda r, dd: (y(r["uD"][dd], w_of(r, dd)) - y(r["uR"], w_of(r, dd))) if informative(r, dd) else None)
out["gap"] = summ(G)
out["cells"] = {}
for cell in CELLS:
    cond, l = cell.split("@")
    l = int(l)
    C = pairwise(lambda r, dd: (y(r["U"][(cond, l, dd)], w_of(r, dd)) - y(r["uR"], w_of(r, dd))) if informative(r, dd) else None)
    out["cells"][cell] = summ(C, G)
    out["cells"][cell]["kl_median"] = float(np.median([r["KL"][(cond, l, dd)] for r in recs for dd in r["dons"]]))
    out["cells"][cell]["same_answer_rate"] = float(np.mean([r["SAME"][(cond, l, dd)] for r in recs for dd in r["dons"]]))


# ------------------------------------------------------------- matched-KL Gaussian correction for (iv)@21
def g_shift(rec, kl, w):
    vals = []
    for j in sorted({k[1] for k in rec["U"] if k[0] == "g"}):
        pts = sorted((max(rec["KL"][k], 1e-12), y(rec["U"][k], w) - y(rec["uR"], w)) for k in rec["U"] if k[0] == "g" and k[1] == j)
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
    if not informative(r, dd):
        return None
    w = w_of(r, dd)
    g = g_shift(r, r["KL"][("iv", 18, dd)], w)
    return None if g is None else y(r["U"][("iv", 18, dd)], w) - y(r["uR"], w) - g


Cc = pairwise(corrected)
out["iv21_kl_corrected"] = summ(Cc, G)
out["iv21_kl_corrected"]["pairs_not_bracketed"] = int(sum(informative(r, dd) and corrected(r, dd) is None for r in recs for dd in r["dons"]))

# ------------------------------------------------------------- input-only and answer-only predictors
items_u, items_feat = {}, {}


def feats(i):
    a, b = (int(v) for v in i[2:].split("*"))
    f = np.zeros(80)
    for k, dg in enumerate([a // 10, a % 10, b // 10, b % 10]):
        f[10 * k + dg] = 1
    for k, ch in enumerate(f"{a * b:04d}"[-4:]):
        f[40 + 10 * k + int(ch)] = 1
    return f


for r in recs:
    items_u.setdefault(r["r"], r["uR"])
    for dd in r["dons"]:
        items_u.setdefault(dd, r["uD"][dd])
keys = sorted(items_u)
X = np.array([feats(i) for i in keys])
Y = np.array([items_u[i] for i in keys])
fold = np.random.default_rng(3).integers(0, 5, len(keys))
pred = np.zeros_like(Y)
for fo in range(5):
    tr, te = fold != fo, fold == fo
    Xt = np.c_[X[tr], np.ones(tr.sum())]
    Wt = np.linalg.solve(Xt.T @ Xt + 1.0 * np.eye(Xt.shape[1]), Xt.T @ Y[tr])
    pred[te] = np.c_[X[te], np.ones(te.sum())] @ Wt
PRED = dict(zip(keys, pred))
ss_res, ss_tot = float(((Y - pred) ** 2).sum()), float(((Y - Y.mean(0)) ** 2).sum())
out["input_predictor"] = dict(cv_r2=1 - ss_res / ss_tot, n_items=len(keys))
Gin = pairwise(lambda r, dd: (y(PRED[dd], w_of(r, dd)) - y(PRED[r["r"]], w_of(r, dd))) if informative(r, dd) else None)
out["input_predicted_gap"] = summ(Gin, G)  # share of the clean donor-recipient report difference that the input predicts
out["input_predicted_cells"] = {}
for cell in CELLS:
    eff_input = "donor" if cell in ("i@6", "ii@6") else "recipient"  # which item's input the patched row effectively carries
    Ci = pairwise(lambda r, dd: ((y(PRED[dd], w_of(r, dd)) - y(PRED[r["r"]], w_of(r, dd))) if eff_input == "donor" else 0.0)
                  if informative(r, dd) else None)
    cond, l = cell.split("@")
    Cx = pairwise(lambda r, dd: (y(r["U"][(cond, int(l), dd)], w_of(r, dd)) - y(r["uR"], w_of(r, dd))
                                 - (((y(PRED[dd], w_of(r, dd)) - y(PRED[r["r"]], w_of(r, dd))) if eff_input == "donor" else 0.0)))
                  if informative(r, dd) else None)
    out["input_predicted_cells"][cell] = dict(predicted=summ(Ci, G), effect_minus_predicted=summ(Cx, G))
out["answer_only_predictor"] = "donor and recipient share the answer tokens and every row is teacher-forced to them: an answer-only predictor predicts no change for any cell (0 by construction)"


# ------------------------------------------------------------- depth transfer in this pass (as arm S2: s-signed far - null)
def snull(val):
    vals_r = []
    num = np.zeros(NB)
    den = np.zeros(NB)
    for r in recs:
        f = [(IX[dd], val(r, dd)) for dd in r["dons"] if r["dons"][dd] == "far"]
        nl = [(IX[dd], val(r, dd)) for dd in r["dons"] if r["dons"][dd] == "null"]
        e = r["s"] * (np.mean([x for _, x in f]) - np.mean([x for _, x in nl]))
        vals_r.append(e)
        wf = np.stack([Wi[:, i] for i, _ in f], 1)
        wn = np.stack([Wi[:, i] for i, _ in nl], 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            eb = r["s"] * ((wf * [x for _, x in f]).sum(1) / wf.sum(1) - (wn * [x for _, x in nl]).sum(1) / wn.sum(1))
        wr = Wi[:, IX[r["r"]]] * np.isfinite(eb)
        num += wr * np.nan_to_num(eb)
        den += wr
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(point=float(np.mean(vals_r)), boot=num / den, n_rec=R, n_pairs=out["pairs_total"])


def Dgen(st):
    return float(np.mean(st[1:])) / L


GD = snull(lambda r, dd: Dgen(r["sD"][dd]) - Dgen(r["sR"]))
out["depth"] = {cell: summ(snull(lambda r, dd, c=cell: Dgen(r["LL"][(c.split("@")[0], int(c.split("@")[1]), dd)]) - Dgen(r["sR"])), GD) for cell in CELLS}
tauD = out["depth"]["iv@18"]["tau"]

# ------------------------------------------------------------- A/B (secondary; replication of ../selfreport)
Gab = snull(lambda r, dd: r["ab"][("D", dd)] - r["ab"]["R"])
out["ab"] = dict(gap=summ(Gab), cells={cell: summ(snull(lambda r, dd, c=cell: r["ab"][(c.split("@")[0], int(c.split("@")[1]), dd)] - r["ab"]["R"]), Gab)
                                       for cell in CELLS},
                 clean_mean_r=float(np.mean([r["ab"]["R"] for r in recs])))


# ------------------------------------------------------------- decision rules (§5)
def moves(c):
    lo, hi = c["ci"]
    return lo is not None and (lo > 0 or hi < 0) and c.get("tau") is not None and abs(c["tau"]) >= 0.2


iv, ivr, i7, ii7 = (out["cells"][k] for k in ("iv@18", "ivrep@18", "i@6", "ii@6"))
gap_ok = out["gap"]["ci"][0] is not None and out["gap"]["ci"][0] > 0
access = (gap_ok and iv["tau"] is not None and iv["tau"] >= 0.5 * tauD and iv["ci"][0] > 0
          and out["iv21_kl_corrected"]["ci"][0] is not None and out["iv21_kl_corrected"]["ci"][0] > 0
          and out["input_predicted_cells"]["iv@18"]["effect_minus_predicted"]["ci"][0] > 0 and not moves(i7))
iv_meets = iv["tau"] is not None and iv["tau"] >= 0.5 * tauD and iv["ci"][0] > 0
if not gap_ok:
    verdict = "NOT EVALUABLE (clean reports do not follow the donor-recipient settling difference: gap CI does not exclude 0 upward)"
elif access:
    verdict = "CAUSAL SELF-ACCESS"
elif (moves(i7) or moves(ii7)) and not iv_meets:
    verdict = "RECONSTRUCTION FROM INPUT"
elif moves(ivr) and not iv_meets:
    verdict = "DIRECT STEERING ONLY"
else:
    verdict = "INCONCLUSIVE"
out["tau_D_iv21"] = tauD
out["verdict"] = verdict
out["inert_check_iir21"] = out["cells"]["iir@18"]["ci"]
json.dump(out, open(os.path.join(RES, "sr2_summary.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ("cells", "input_predicted_cells", "depth", "ab")}, indent=1, default=float))
for k, v in out["cells"].items():
    print(f"{k:9s} effect {v['effect']:+.4f} {[round(x, 4) for x in v['ci']]} tau {v['tau']:+.2f} {[round(x, 2) for x in v['tau_ci']]} "
          f"| depth tau {out['depth'][k]['tau']:+.2f} | AB effect {out['ab']['cells'][k]['effect']:+.3f} {[round(x, 3) for x in out['ab']['cells'][k]['ci']]} "
          f"| kl~{v['kl_median']:.1f} same {v['same_answer_rate']:.3f}")
print("AB gap", out["ab"]["gap"])
