"""Paper 8 patching arm 2: effects, two-way item-bootstrap CIs, controls and decision rules
(patch2/PREDICTIONS.md §4-§6).

python analyze_patch2.py [smoke]

Effect of a cell (condition, layer) on a measure X: for each recipient r, s_r * (mean X over its far-donor
rows - mean X over its null-donor rows), s_r = +1 for an early recipient (late donors), -1 for a late one;
averaged over recipients. Full-transfer reference ("gap"): the same contrast on the in-batch clean donor
rows. CIs: two-way item bootstrap (one Poisson(1) weight per item, used both when it is a recipient and when
it is a donor; 10,000 resamples, percentile 95%); a recipient-only bootstrap is also reported. Derived
contrasts (differences, interaction, share) use the same resamples. Writes res/patch2_summary.json.
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
PRIMARY = [14, 21]
CONDS = ["i", "ii", "iii", "iv", "iip", "iiip"]
NAME = dict(i="(i)", ii="(ii)", iii="(iii)", iv="(iv)", iip="(ii+)", iiip="(iii+)")
NB = 10000
rng = np.random.default_rng(80)


def settle(t):
    """paper 4's settle_top1 on a list: smallest l with top1[k] == top1[-1] for every k >= l."""
    l = len(t) - 1
    while l > 0 and t[l - 1] == t[-1]:
        l -= 1
    return l


def mean_or_nan(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else np.nan


# ------------------------------------------------------------------ measures
# fn(settle list, n_dig, layer) -> value; applied identically to patched rows and to clean donor rows
def m_D(st, nd, l):
    return np.mean(st) / L if st else np.nan


def m_Dgen(st, nd, l):
    return np.mean(st[1:]) / L if len(st) > 1 else np.nan


def m_Dgen_above(st, nd, l):  # lens-copy-free: only layers > l can lower a digit's settle layer
    return np.mean([max(s, l + 1) for s in st[1:]]) / L if len(st) > 1 else np.nan


def m_digit(k):
    def f(st, nd, l):
        return float(st[k]) if nd == 4 and len(st) == 4 else np.nan  # in layers
    return f


MEASURES = dict(D=m_D, D_gen=m_Dgen, D_gen_above=m_Dgen_above, **{f"digit{k + 1}": m_digit(k) for k in range(4)})


def load():
    p = os.path.join(RES, "patchS2" + ("_smoke" if SMOKE else "") + ".jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


recs = load()
assert recs, "no records"
ids = sorted({r["r"] for r in recs} | {d for r in recs for d in r["donclean"]})
IX = {i: k for k, i in enumerate(ids)}
R = len(recs)
Wi = rng.poisson(1.0, (NB, len(ids))).astype(float)
Wr = rng.multinomial(R, np.ones(R) / R, NB).astype(float)
S_ = np.array([r["s"] for r in recs], float)
IR = np.array([IX[r["r"]] for r in recs])


def arrays(val):
    """val(rec, role, k_th donor of that role) -> value. Returns far/null matrices and donor indices."""
    out = {}
    for role in ("far", "null"):
        lists = [[d for d in r["donclean"] if d in r["roles"][role]] for r in recs]
        Kmax = max(len(x) for x in lists)
        V = np.full((R, Kmax), np.nan)
        I = np.zeros((R, Kmax), int)
        for a, (r, ds) in enumerate(zip(recs, lists)):
            for k, d in enumerate(ds):
                V[a, k], I[a, k] = val(r, d), IX[d]
        out[role] = (V, I)
    return out


for r in recs:
    r["roles"] = dict(far={x["donor"] for x in r["rows"] if x["role"] == "far"},
                      null={x["donor"] for x in r["rows"] if x["role"] == "null"})
    r["cell"] = {}
    for x in r["rows"]:
        r["cell"][(x["cond"], x["l"], x["donor"])] = x


def wmean(V, W):
    ok = ~np.isnan(V)
    W = W * ok
    sw = W.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sw > 0, (W * np.nan_to_num(V)).sum(-1) / sw, np.nan)


def contrast(A):
    """Point estimate and the two bootstrap distributions (two-way item, recipient-only)."""
    (F, iF), (N, iN) = A["far"], A["null"]
    with np.errstate(invalid="ignore"):
        e = S_ * (np.nanmean(F, 1) - np.nanmean(N, 1))
    ok = np.isfinite(e)
    point = float(np.mean(e[ok]))
    f = wmean(F[None], Wi[:, iF])
    n = wmean(N[None], Wi[:, iN])
    eb = S_[None] * (f - n)
    w = Wi[:, IR] * np.isfinite(eb)
    with np.errstate(invalid="ignore", divide="ignore"):
        two = (w * np.nan_to_num(eb)).sum(1) / w.sum(1)
    wr = Wr * ok[None]
    rec = (wr * np.nan_to_num(e)[None]).sum(1) / wr.sum(1)
    return dict(point=point, two=two, rec=rec, n=int(ok.sum()))


def ci(x):
    x = x[np.isfinite(x)]
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def classify(E, c, sesoi=SESOI):
    if E >= sesoi and c[0] > 0:
        return "moves"
    if E <= -sesoi and c[1] < 0:
        return "moves (wrong sign)"
    if c[0] > -sesoi and c[1] < sesoi:
        return "no meaningful move" + (" (small, nonzero)" if c[0] > 0 or c[1] < 0 else "")
    return "inconclusive"


def patched_val(meas, cond, l):
    return lambda r, d: (lambda x: MEASURES[meas](x["settle"], x["n_dig"], l) if x is not None else np.nan)(
        r["cell"].get((cond, l, d)))


def clean_val(meas, l):
    return lambda r, d: MEASURES[meas](r["donclean"][d]["settle"], r["donclean"][d]["n_dig"], l)


def splice_val(meas, l):
    """The exact-interchange expectation: recipient's clean lens below l, the donor row's from l up."""
    def f(r, d):
        st = [settle(a[:l] + b[l:]) for a, b in zip(r["clean"]["top1"], r["donclean"][d]["top1"])]
        return MEASURES[meas](st, len(st), l)
    return f


def summarise(C, gap, sesoi=SESOI):
    d = dict(effect=C["point"], ci=ci(C["two"]), ci_recipient=ci(C["rec"]), n_recipients=C["n"],
             cls=classify(C["point"], ci(C["two"]), sesoi), gap=gap)
    d["tau"] = C["point"] / gap if gap else None
    d["tau_ci"] = [x / gap for x in d["ci"]] if gap else None
    return d


def per_rec_rate(cond, l, fn, role=None):
    v = []
    for r in recs:
        xs = [fn(x, r) for x in r["rows"] if x["cond"] == cond and x["l"] == l and (role is None or x["role"] == role)]
        if xs:
            v.append(np.mean(xs))
    v = np.array(v)
    bs = v[rng.integers(0, len(v), (NB, len(v)))].mean(1)
    return dict(mean=float(v.mean()), ci=ci(bs), n=len(v))


# ------------------------------------------------------------------ cells
cells, boot = {}, {}
gaps = {}
for l in LAYERS:
    for meas in MEASURES:
        gaps[(meas, l)] = contrast(arrays(clean_val(meas, l)))["point"]
for cond in CONDS:
    for l in LAYERS:
        c = {}
        for meas in MEASURES:
            C = contrast(arrays(patched_val(meas, cond, l)))
            boot[(cond, l, meas)] = C
            c[meas] = summarise(C, gaps[(meas, l)], 1.0 if meas.startswith("digit") else SESOI)
        for nm, sg in (("early_recipients", 1), ("late_recipients", -1)):
            e = [r["s"] * (mean_or_nan([r["cell"][(cond, l, d)]["D"] for d in r["roles"]["far"]])
                           - mean_or_nan([r["cell"][(cond, l, d)]["D"] for d in r["roles"]["null"]]))
                 for r in recs if r["s"] == sg]
            c[nm] = dict(effect=float(np.nanmean(e)), n=len(e))
        c["same_tokens"] = per_rec_rate(cond, l, lambda x, r: x["same_tok"])
        c["changed_far"] = per_rec_rate(cond, l, lambda x, r: 1 - x["same_tok"], "far")
        c["changed_null"] = per_rec_rate(cond, l, lambda x, r: 1 - x["same_tok"], "null")
        c["correct"] = per_rec_rate(cond, l, lambda x, r: x["ok"])
        cells[f"{cond}|{l}"] = c

# exact-interchange expectation (the ceiling an all-position patch can reach, given the lens below l)
ceiling = {}
for l in LAYERS:
    ceiling[l] = {}
    for meas in ("D", "D_gen"):
        C = contrast(arrays(splice_val(meas, l)))
        ceiling[l][meas] = dict(effect=C["point"], tau=C["point"] / gaps[(meas, l)])
    match = [x["settle"] == [settle(a[:l] + b[l:]) for a, b in zip(r["clean"]["top1"], r["donclean"][x["donor"]]["top1"])]
             for r in recs for x in r["rows"] if x["cond"] == "iiip" and x["l"] == l]
    ceiling[l]["iiip_rows_equal_splice"] = float(np.mean(match))


def derived(name, terms, meas, l, sesoi=SESOI):
    """Linear combination sum(w * effect(cond)) on the same resamples."""
    pt = sum(w * boot[(cd, l, meas)]["point"] for cd, w in terms)
    two = sum(w * boot[(cd, l, meas)]["two"] for cd, w in terms)
    c2 = ci(two)
    return dict(name=name, effect=pt, ci=c2, cls=classify(pt, c2, sesoi))


V = {}
# ---- PC1 (§4): the corrected ceiling, lenient single-cell rule at each of 7, 14, 21 (24 gates layer 24 only)
pc1 = {}
for l in LAYERS:
    c = cells[f"iiip|{l}"]
    ok = (c["D"]["tau"] >= 0.8 * ceiling[l]["D"]["tau"] and c["D"]["ci"][0] > 0 and c["same_tokens"]["mean"] >= 0.90)
    pc1[l] = dict(passed=bool(ok), tau=c["D"]["tau"], tau_exact=ceiling[l]["D"]["tau"], ci=c["D"]["ci"],
                  same_tokens=c["same_tokens"]["mean"], rows_equal_splice=ceiling[l]["iiip_rows_equal_splice"])
V["PC1"] = all(pc1[l]["passed"] for l in (7, 14, 21))
V["PC1_24"] = pc1[24]["passed"]
V["PC1_detail"] = pc1

# ---- primary (§5.1): is depth held at the answer positions?
held = {}
for l in LAYERS:
    c = cells[f"iv|{l}"]
    g, a = c["D_gen"], c["D_gen_above"]
    if g["cls"] == "moves" and a["ci"][0] > 0:
        lab = "HELD"
    elif g["cls"].startswith("no meaningful move"):
        lab = "NOT HELD" + (" (small, nonzero)" if "nonzero" in g["cls"] else "")
    elif g["cls"] == "moves (wrong sign)":
        lab = "WRONG SIGN"
    elif g["ci"][0] > 0:
        lab = "PARTIAL"
    else:
        lab = "INCONCLUSIVE"
    sh_two = boot[("iv", l, "D_gen")]["two"] / boot[("iiip", l, "D_gen")]["two"]
    held[l] = dict(label=lab, D_gen=g["effect"], ci=g["ci"], tau_gen=g["tau"], tau_gen_ci=g["tau_ci"], cls=g["cls"],
                   copy_free=dict(effect=a["effect"], ci=a["ci"], tau=a["tau"]),
                   share_of_ceiling=dict(point=g["effect"] / cells[f"iiip|{l}"]["D_gen"]["effect"], ci=ci(sh_two)))
V["held"] = held
if not V["PC1"]:
    V["verdict"] = "INVALID"
else:
    V["verdict"] = "; ".join(f"answer positions @{l}: {held[l]['label']}" for l in PRIMARY)
    if not V["PC1_24"]:
        V["verdict"] += " (layer 24 readouts void: PC1-24 failed)"

# ---- secondary (§5.2)
sec = {}
for l in LAYERS:
    sec[l] = dict(
        interaction=derived("(iii+) - (iii) - (iv)", [("iiip", 1), ("iii", -1), ("iv", -1)], "D_gen", l),
        rest_of_prompt=derived("(iii+) - (ii+)", [("iiip", 1), ("iip", -1)], "D", l),
        prefill_given_answer=derived("(ii+) - (iv)", [("iip", 1), ("iv", -1)], "D_gen", l),
        answer_given_prompt=derived("(iii+) - (iii)", [("iiip", 1), ("iii", -1)], "D_gen", l),
    )
V["secondary"] = sec
# causality check: (iv) cannot change digit 1 (read at the last prompt position, before any answer position)
d1 = [x["settle"][0] == r["clean"]["settle"][0] for r in recs for x in r["rows"] if x["cond"] == "iv" and x["settle"]]
V["iv_digit1_unchanged"] = float(np.mean(d1))

summary = dict(n_recipients=R, n_items=len(ids), cells=cells, ceiling=ceiling,
               gap_pool_D=float(np.mean([r["s"] * (np.mean([r["donD"][d] for d in r["roles"]["far"]])
                                                    - np.mean([r["donD"][d] for d in r["roles"]["null"]])) for r in recs])),
               gap_inbatch={f"{k[0]}|{k[1]}": v for k, v in gaps.items()},
               clean_D_equals_pool=float(np.mean([r["clean"]["D"] == r["D_pool"] for r in recs])),
               donor_D_equals_pool=float(np.mean([r["donclean"][d]["D"] == r["donD"][d] for r in recs for d in r["donclean"]])),
               donor_argmax_is_shared_answer=float(np.mean([r["donclean"][d]["cont_same"] for r in recs for d in r["donclean"]])),
               clean_cont_same=float(np.mean([r["clean"]["cont_same"] for r in recs])),
               chunk_mismatch=int(sum(r["clean"].get("chunk_mismatch", 0) for r in recs)
                                  + sum(v.get("chunk_mismatch", 0) for r in recs for v in r["donclean"].values())),
               verdict=V)
json.dump(summary, open(os.path.join(RES, "patch2" + ("_smoke" if SMOKE else "") + "_summary.json"), "w"), indent=1)


# ------------------------------------------------------------------ print
def fmt(c, k=4):
    return f"{c['effect']:+.{k}f} [{c['ci'][0]:+.{k}f}, {c['ci'][1]:+.{k}f}]"


print(f"arm S2: {R} recipients, {len(ids)} items; in-batch gap D {gaps[('D', 7)] * L:.3f} layers, D_gen "
      f"{gaps[('D_gen', 7)] * L:.3f}; pool gap D {summary['gap_pool_D'] * L:.3f}")
print(f"clean D = pool {summary['clean_D_equals_pool']:.2f}; donor D = pool {summary['donor_D_equals_pool']:.2f}; "
      f"donor argmax = shared answer {summary['donor_argmax_is_shared_answer']:.3f}; chunk mismatches {summary['chunk_mismatch']}")
print(f"\n{'cell':10s} {'D effect [95% CI]':30s} {'class':30s} {'tau':>5s}  {'D_gen effect':30s} {'tau_g':>5s}  "
      f"{'D_gen>l':28s} {'same':>5s}")
for cond in CONDS:
    for l in LAYERS:
        c = cells[f"{cond}|{l}"]
        print(f"{NAME[cond] + '@' + str(l):10s} {fmt(c['D']):30s} {c['D']['cls']:30s} {c['D']['tau']:5.2f}  "
              f"{fmt(c['D_gen']):30s} {c['D_gen']['tau']:5.2f}  {fmt(c['D_gen_above']):28s} {c['same_tokens']['mean']:5.2f}")
print("\nper-digit effects (layers; gap in parentheses)")
for cond in CONDS:
    for l in LAYERS:
        c = cells[f"{cond}|{l}"]
        print(f"{NAME[cond] + '@' + str(l):10s} " + "  ".join(
            f"d{k + 1} {fmt(c[f'digit{k + 1}'], 2)} ({c[f'digit{k + 1}']['gap']:.2f})" for k in range(4)))
print("\nceiling (exact splice):", {l: {m_: round(v["tau"], 3) for m_, v in ceiling[l].items() if isinstance(v, dict)}
                                 for l in LAYERS})
print("\nverdict:", json.dumps(V, indent=1))
