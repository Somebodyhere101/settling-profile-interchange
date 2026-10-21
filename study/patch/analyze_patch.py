"""Paper 8 patching arm: effects, item-bootstrap CIs, decision rules (patch/PREDICTIONS.md §4-§5).

python analyze_patch.py [smoke]

Effect of a cell (arm, condition, layer): for each recipient r, s_r * (mean D over its far-donor rows -
mean D over its null-donor rows), s_r = +1 for an early recipient (late donors), -1 for a late one;
averaged over recipients. CIs: two-way item bootstrap (Poisson(1) weight per item, used for the item
both as recipient and as donor; 10,000 resamples, percentile 95%); a recipient-only bootstrap is also
reported. Writes res/patch_summary.json.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
L = 28
SESOI = 1 / L
LAYERS = [7, 14, 21, 24]
PILOT_LAYERS = [14, 21, 24]
CONDS = ["i", "ii", "iir", "iii"]
NB = 10000
rng = np.random.default_rng(80)


def load(arm):
    p = os.path.join(RES, f"patch{arm}" + ("_smoke" if SMOKE else "") + ".jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def cell_arrays(recs, cond, l, key):
    """Per recipient: far and null values of `key` (NaN if missing), donor item indices, sign."""
    ids = sorted({r["r"] for r in recs} | {x["donor"] for r in recs for x in r["rows"]})
    ix = {i: k for k, i in enumerate(ids)}
    KF = max(sum(1 for x in r["rows"] if x["role"] == "far" and x["cond"] == cond and x["l"] == l) for r in recs)
    KN = max(sum(1 for x in r["rows"] if x["role"] == "null" and x["cond"] == cond and x["l"] == l) for r in recs)
    R = len(recs)
    F, N = np.full((R, KF), np.nan), np.full((R, KN), np.nan)
    iF, iN = np.zeros((R, KF), int), np.zeros((R, KN), int)
    s, ir = np.zeros(R), np.zeros(R, int)
    for a, r in enumerate(recs):
        s[a], ir[a] = r["s"], ix[r["r"]]
        f = [x for x in r["rows"] if x["role"] == "far" and x["cond"] == cond and x["l"] == l]
        nn = [x for x in r["rows"] if x["role"] == "null" and x["cond"] == cond and x["l"] == l]
        for k, x in enumerate(f):
            F[a, k], iF[a, k] = (np.nan if x[key] is None else x[key]), ix[x["donor"]]
        for k, x in enumerate(nn):
            N[a, k], iN[a, k] = (np.nan if x[key] is None else x[key]), ix[x["donor"]]
    return dict(F=F, N=N, iF=iF, iN=iN, s=s, ir=ir, n_items=len(ids))


def wmean(V, W):
    ok = ~np.isnan(V)
    W = W * ok
    sw = W.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, (W * np.nan_to_num(V)).sum(-1) / sw, np.nan)


def contrast(A, Wi=None, Wr=None):
    """Mean over recipients of s * (far mean - null mean). Wi: item weights [nb, n_items] (two-way);
    Wr: recipient weights [nb, R] (recipient-only). Returns point estimate or [nb] resamples."""
    if Wi is None and Wr is None:
        e = A["s"] * (np.nanmean(A["F"], 1) - np.nanmean(A["N"], 1))
        return float(np.nanmean(e)), int(np.isfinite(e).sum())
    if Wi is not None:
        f = wmean(A["F"][None], Wi[:, A["iF"]])
        n = wmean(A["N"][None], Wi[:, A["iN"]])
        wr = Wi[:, A["ir"]]
    else:
        f = np.nanmean(A["F"], 1)[None]
        n = np.nanmean(A["N"], 1)[None]
        wr = Wr
    e = A["s"][None] * (f - n)
    ok = np.isfinite(e)
    wr = wr * ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return (wr * np.nan_to_num(e)).sum(1) / wr.sum(1)


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def classify(E, c):
    if E >= SESOI and c[0] > 0:
        return "moves"
    if E <= -SESOI and c[1] < 0:
        return "moves (wrong sign)"
    if c[0] > -SESOI and c[1] < SESOI:
        return "no meaningful move" + (" (small, nonzero)" if c[0] > 0 or c[1] < 0 else "")
    return "inconclusive"


def per_recipient_mean(recs, cond, l, fn):
    """Mean over a recipient's rows (far and null) of fn(row, rec), then the mean over recipients, with a
    two-way-free recipient bootstrap CI (content measures)."""
    v = []
    for r in recs:
        xs = [fn(x, r) for x in r["rows"] if x["cond"] == cond and x["l"] == l]
        xs = [x for x in xs if x is not None]
        if xs:
            v.append(np.mean(xs))
    v = np.array(v)
    if not len(v):
        return None
    bs = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return dict(mean=float(v.mean()), ci=ci(bs), n=len(v))


def analyse(arm, recs):
    out = {}
    ids = sorted({r["r"] for r in recs} | {x["donor"] for r in recs for x in r["rows"]})
    Wi = rng.poisson(1.0, (NB, len(ids))).astype(float)
    Wr = rng.multinomial(len(recs), np.ones(len(recs)) / len(recs), NB).astype(float)
    for cond in CONDS:
        for l in LAYERS:
            c = {}
            for key in ("D", "D_gen"):
                A = cell_arrays(recs, cond, l, key)
                E, nr = contrast(A)
                c2 = ci(contrast(A, Wi=Wi))
                c[key] = dict(effect=E, ci=c2, ci_recipient=ci(contrast(A, Wr=Wr)), n_recipients=nr,
                              cls=classify(E, c2))
            # full-transfer reference: the same contrast on the donors' clean D
            g = np.array([r["s"] * (np.mean([r["donD"][d] for d in {x["donor"] for x in r["rows"] if x["role"] == "far"}])
                                    - np.mean([r["donD"][d] for d in {x["donor"] for x in r["rows"] if x["role"] == "null"}]))
                          for r in recs])
            c["gap"] = float(g.mean())
            c["tau"] = c["D"]["effect"] / c["gap"] if c["gap"] else None
            c["tau_ci"] = [x / c["gap"] for x in c["D"]["ci"]]
            # by recipient direction
            for nm, sg in (("early_recipients", 1), ("late_recipients", -1)):
                sub = [r for r in recs if r["s"] == sg]
                if sub:
                    A = cell_arrays(sub, cond, l, "D")
                    c[nm] = dict(effect=contrast(A)[0], n=len(sub))
            # content and disruption
            c["don_ans"] = per_recipient_mean(recs, cond, l, lambda x, r: x["don_ans"])
            c["changed"] = per_recipient_mean(recs, cond, l, lambda x, r: int(x["ans"] != r["clean"]["ans"]))
            c["correct_for_recipient"] = per_recipient_mean(recs, cond, l, lambda x, r: x["ok"])
            for role in ("far", "null"):
                c[f"changed_{role}"] = per_recipient_mean(recs, cond, l, lambda x, r, ro=role:
                                                          int(x["ans"] != r["clean"]["ans"]) if x["role"] == ro else None)
            if arm == "P":
                c["dlm"] = per_recipient_mean(recs, cond, l, lambda x, r: x["dlm"])
            out[f"{cond}|{l}"] = c
    clean_vs_pool = float(np.mean([r["clean"]["D"] == r["D_pool"] for r in recs]))
    mism = int(sum(r["clean"].get("chunk_mismatch", 0) for r in recs))
    return dict(cells=out, n_recipients=len(recs), n_items=len(ids), clean_D_equals_pool=clean_vs_pool,
                chunk_clean_mismatch=mism)


def fmt(c):
    return f"{c['effect']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"


summary = {}
for arm in ("P", "S"):
    recs = load(arm)
    if recs:
        summary[arm] = analyse(arm, recs)

V = {}
if "P" in summary:
    C = summary["P"]["cells"]
    # PC1: the all-position patch transfers depth and answer at layers 7, 14, 21
    pc1 = {l: dict(D=C[f"iii|{l}"]["D"]["cls"], don_ans=C[f"iii|{l}"]["don_ans"]["mean"]) for l in (7, 14, 21)}
    V["PC1"] = all(v["D"] == "moves" and v["don_ans"] >= 0.90 for v in pc1.values())
    V["PC1_detail"] = pc1
    # PC2: the question-token patch at layer 7 moves the log-odds of the donor's answer by >= 0.5 nats
    d = C["i|7"]["dlm"]
    V["PC2"] = bool(d["mean"] >= 0.5 and d["ci"][0] > 0)
    V["PC2_detail"] = d
    mv = {f"{c}|{l}": C[f"{c}|{l}"]["D"]["cls"] for c in ("i", "ii") for l in PILOT_LAYERS}
    A_cells = [k for k, v in mv.items() if v == "moves"]
    caveat = [k for k in A_cells if k.startswith("ii|") and not C[k]["D_gen"]["ci"][0] > 0]
    if not V["PC1"]:
        verdict = "INVALID"
    elif A_cells:
        verdict = "A" + ("-source" if any(k.startswith("i|") for k in A_cells) else "") + \
                  ("-reading" if any(k.startswith("ii|") for k in A_cells) else "")
    elif any(v == "moves (wrong sign)" for v in mv.values()):
        verdict = "OTHER"
    else:
        verdict = "B-strong" if all(v.startswith("no meaningful move") for v in mv.values()) else "B-weak"
    if verdict.startswith("B") and not V["PC2"]:
        verdict += " ((i) uninformative: no answer content from the question tokens)"
    V.update(verdict=verdict, cells_pilot_layers=mv, A_cells=A_cells, lens_copy_caveat=caveat)
    # secondary (§5.2): substantial partial transfer that the SESOI rule cannot call "moves"
    V["partial_transfer_cells"] = [k for k in mv if C[k]["D"]["ci"][0] > 0 and C[k]["tau"] >= 0.5 and k not in A_cells]
    V["S1_reading_position_only"] = {l: C[f"iir|{l}"]["D"]["cls"] for l in PILOT_LAYERS}
    if "S" in summary:
        CS = summary["S"]["cells"]
        q = {}
        for k in A_cells:
            e = CS[k]["D"]
            if e["ci"][0] > 0:
                q[k] = "answer-independent"
            elif e["cls"].startswith("no meaningful move"):
                q[k] = "answer-bound"
            else:
                q[k] = "inconclusive"
        V["S2_qualifier"] = q
summary["verdict"] = V
json.dump(summary, open(os.path.join(RES, "patch" + ("_smoke" if SMOKE else "") + "_summary.json"), "w"), indent=1)

for arm in summary:
    if arm == "verdict":
        continue
    S_ = summary[arm]
    print(f"\n=== arm {arm}: {S_['n_recipients']} recipients, {S_['n_items']} items; in-batch clean D = pool D "
          f"for {S_['clean_D_equals_pool']:.2f}; chunk mismatches {S_['chunk_clean_mismatch']}")
    print(f"{'cell':8s} {'D effect [95% CI]':32s} {'class':32s} {'tau':>6s} {'D_gen effect':30s} "
          f"{'don_ans':>7s} {'chg f/n':>9s}" + ("  dlm (nats)" if arm == "P" else ""))
    for cond in CONDS:
        for l in LAYERS:
            c = S_["cells"][f"{cond}|{l}"]
            line = (f"{cond + '|' + str(l):8s} {fmt(c['D']):32s} {c['D']['cls']:32s} {c['tau']:6.2f} "
                    f"{fmt(c['D_gen']):30s} {c['don_ans']['mean']:7.2f} "
                    f"{c['changed_far']['mean']:.2f}/{c['changed_null']['mean']:.2f}")
            if arm == "P":
                line += f"  {c['dlm']['mean']:+.2f} [{c['dlm']['ci'][0]:+.2f}, {c['dlm']['ci'][1]:+.2f}]"
            print(line)
print("\nverdict:", json.dumps(V, indent=1))
