"""Paper 8 pilot analysis (decision rules in PREDICTIONS.md §5).

python analyze8.py <model key> [smoke]  ->  res/<key>_summary.json and a printed table

Per target (settle, flip): readability of o and os at each layer (Check A form), the matched-KL push
effect of every (condition, layer, kind, strength, sign) cell against the Gaussian reference pushed at
the same positions and layer (paper 6 analyze.py check_D, "rand" family: per-item linear interpolation
in log KL, two-level bootstrap over items and reference directions), the norm-matched effect, the
persistence of the writes, and the hypothesis verdicts.
"""

import json
import math
import os
import sys

import numpy as np
from scipy.stats import rankdata, spearmanr

KEY = sys.argv[1]
SMOKE = len(sys.argv) > 2 and sys.argv[2] == "smoke"
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "res")
CFG = json.load(open(os.path.join(HERE, "config8.json"), encoding="utf-8"))
NB = 2000
K = CFG["k"]
OFF = CFG["persist_offset"]


def load(target):
    pre = os.path.join(RES, f"{target}_{KEY}" + ("_smoke" if SMOKE else ""))
    if not os.path.exists(pre + "_push.jsonl"):
        return None
    info = json.load(open(pre + "_info.json", encoding="utf-8"))
    clean = [json.loads(l) for l in open(pre + "_clean.jsonl", encoding="utf-8")]
    push = [json.loads(l) for l in open(pre + "_push.jsonl", encoding="utf-8")]
    H = dict(np.load(pre + "_clean_h.npz"))
    dirs = dict(np.load(pre + "_dirs.npz"))
    return info, clean, push, H, dirs


# ---------------------------------------------------------------- statistics (paper 6 analyze.py)
def auc_fast(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    n1, n0 = y.sum(), len(y) - y.sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def partial_rho(x, y, z):
    z = np.asarray(z, float)
    z = z[:, None] if z.ndim == 1 else z
    rx, ry = rankdata(x), rankdata(y)
    rz = np.column_stack([rankdata(c) for c in z.T])

    def res(a, b):
        b1 = np.c_[np.ones(len(a)), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    ex, ey = res(rx, rz), res(ry, rz)
    d = np.sqrt((ex @ ex) * (ey @ ey))
    return float(ex @ ey / d) if d > 0 else float("nan")


def boot(fn, n, seed=0):
    rng = np.random.default_rng(seed)
    out = [fn(rng.integers(0, n, n)) for _ in range(NB)]
    out = np.array([o for o in out if o is not None and not (isinstance(o, float) and math.isnan(o))])
    if len(out) < NB * 0.5:
        return [None, None]
    return [float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))]


def interp_ref(kl_push, pts):
    """paper 6: linear interpolation of dT in log KL between the two grid norms that bracket kl_push."""
    pts = sorted((max(k, 1e-12), v) for k, v in pts if k is not None and v is not None)
    if not pts or kl_push is None:
        return None
    lk = math.log(max(kl_push, 1e-12))
    xs = [math.log(k) for k, _ in pts]
    if lk < xs[0] or lk > xs[-1]:
        return None
    for (x0, (k0, v0)), (x1, (k1, v1)) in zip(zip(xs, pts), zip(xs[1:], pts[1:])):
        if x0 <= lk <= x1:
            return v0 if x1 == x0 else v0 + (v1 - v0) * (lk - x0) / (x1 - x0)
    return None


# ---------------------------------------------------------------- readability (Check A form)
def readability(info, clean, H, dirs, binary):
    A_ids = set(x["id"] for x in A_SETS[info["target"]])
    A = [r for r in clean if r["id"] in A_ids and r["T"] is not None]
    T = np.array([r["T"] for r in A], float)
    npr = np.array([r["n_prompt"] for r in A], float)
    rho_len = float(spearmanr(T, npr)[0]) if len(set(T)) > 1 and len(set(npr)) > 1 else float("nan")
    use_partial = abs(rho_len) > 0.1

    def stat(P, idx):
        t, p, z = T[idx], P[idx], npr[idx]
        if len(set(t.tolist())) < 2:
            return None
        if use_partial:
            return partial_rho(p, t, z)
        return auc_fast(p, t) if binary else float(spearmanr(p, t)[0])
    ses, null = (0.20, 0.0) if (use_partial or not binary) else (0.60, 0.5)
    out = dict(n=len(A), T_mean=float(T.mean()), rho_T_promptlen=rho_len,
               primary="partial_rho" if use_partial else ("auc" if binary else "rho"))
    for l in info["layers"]:
        for kind, suf in (("o", "r"), ("os", "s")):
            P = np.array([H[f"{r['id']}|{l}|{suf}"] for r in A], float) @ dirs[f"{kind}_{l}"]
            idx = np.arange(len(T))
            est = stat(P, idx)
            ci = boot(lambda i: stat(P, i), len(T))
            v = "inconclusive"
            if est is not None and ci[0] is not None:
                v = "pass" if (ci[0] > null and est >= ses) else ("fail" if ci[1] < ses else "inconclusive")
            out[f"{kind}_{l}"] = dict(est=est, ci=ci, verdict=v)
    return out


# ---------------------------------------------------------------- matched-KL cells
def cell(push, cond, l, kind, k, sign, sesoi):
    items = []
    for rec in push:
        rows = rec["rows"]
        t0 = rows[0]["T"]
        if t0 is None:
            continue
        hit = [r for r in rows if r["kind"] == kind and r["cond"] == cond and r["l"] == l and r["k"] == sign * k]
        if not hit or hit[0]["T"] is None:
            continue
        kl, dt = hit[0]["kl"], hit[0]["T"] - t0
        ref = {}
        for r in rows:
            if r["kind"] == "rand" and r["cond"] == cond and r["l"] == l and r["T"] is not None:
                ref.setdefault(r["j"], []).append((r["kl"], r["T"] - t0, r["c"]))
        refs = {j: interp_ref(kl, [(a, b) for a, b, _ in v]) for j, v in ref.items()}
        norm_ref = np.mean([b for v in ref.values() for a, b, c in v if c == k] or [np.nan])
        items.append(dict(id=rec["id"], kl=kl, dt=dt, refs=refs, norm_ref=norm_ref))
    if not items:
        return None
    out = dict(n_items=len(items))
    R = sorted({j for x in items for j in x["refs"]})
    M = np.array([[x["refs"].get(j) if x["refs"].get(j) is not None else np.nan for j in R] for x in items], float)
    dts = np.array([x["dt"] for x in items], float)
    ok = ~np.all(np.isnan(M), 1)
    out["n_excluded"] = int((~ok).sum())
    dts, M = dts[ok], M[ok]
    out["kl_median"] = float(np.median([x["kl"] for x in items]))
    if len(dts) == 0:
        out.update(n=0, verdict="not_bracketed", controllable=False)
        return out
    est = float(np.mean(dts - np.nanmean(M, 1)))
    rng = np.random.default_rng(1)
    two, one = [], []
    for _ in range(NB):
        ii = rng.integers(0, len(dts), len(dts))
        jj = rng.integers(0, M.shape[1], M.shape[1])
        with np.errstate(all="ignore"):
            a = np.nanmean(M[np.ix_(ii, jj)], 1)
            b = np.nanmean(M[ii], 1)
        if not np.all(np.isnan(a)):
            two.append(np.nanmean(dts[ii] - a))
        one.append(np.nanmean(dts[ii] - b))
    ci = [float(np.quantile(two, 0.025)), float(np.quantile(two, 0.975))]
    out.update(n=int(len(dts)), effect=est, ci=ci, ci_item_only=[float(np.quantile(one, 0.025)), float(np.quantile(one, 0.975))],
               raw_dT=float(np.mean(dts)), mean_ref=float(np.nanmean(M)))
    nr = np.array([x["dt"] - x["norm_ref"] for x in items if not np.isnan(x["norm_ref"])])
    out["norm_matched"] = dict(effect=float(nr.mean()) if len(nr) else None,
                               ci=boot(lambda i: float(nr[i].mean()), len(nr)) if len(nr) else [None, None])
    # paper 6's verdict in the claimed sign (+o claims +T) ...
    e, (lo, hi) = est * sign, sorted([c * sign for c in ci])
    out["verdict"] = "pass" if (lo > 0 and e >= sesoi) else ("reversal" if hi < 0 else ("fail" if hi < sesoi else "inconclusive"))
    # ... and PREDICTIONS.md §5's sign-free criterion
    out["controllable"] = bool(abs(est) >= sesoi and (ci[0] > 0 or ci[1] < 0))
    out["direction"] = int(np.sign(est))
    return out


def moves(cells, cond, kind, layers):
    """PREDICTIONS.md §5: at some layer and sign, two adjacent strengths (1,2 or 2,4) are both
    controllable with the same direction of effect."""
    hits = []
    for l in layers:
        for s in (1, -1):
            for k1, k2 in ((1, 2), (2, 4)):
                a, b = cells.get(f"{kind}|{cond}|{l}|{k1}|{s}"), cells.get(f"{kind}|{cond}|{l}|{k2}|{s}")
                if a and b and a.get("controllable") and b.get("controllable") and a["direction"] == b["direction"]:
                    hits.append(dict(layer=l, sign=s, k=[k1, k2], effects=[a["effect"], b["effect"]]))
    return dict(moves=bool(hits), hits=hits)


# ---------------------------------------------------------------- persistence
def persistence(info, push):
    """Survival of the written component at the answer positions (P1 reading position, P2 last prompt
    token), oracle minus norm-matched Gaussian, at layer l + OFF (primary) and at the last layer."""
    L = info["L"]
    out = {}
    for cond in ("a", "b", "c"):
        for l in info["layers"]:
            for k in K:
                for kind in (("o", "os") if cond == "c" else ("o",)):
                    for tag, li in (("plus%d" % OFF, min(OFF, L - 1 - l)), ("last", L - 1 - l)):
                        for pi, pname in ((0, "P1"), (1, "P2")):
                            d, so, sr = [], [], []
                            for rec in push:
                                p = rec["persist"]
                                ks = [f"{kind}|{cond}|{l}|{s * k}" for s in (1, -1)]
                                kr = f"rand|{cond}|{l}|{k}"
                                if kr not in p or not all(x in p for x in ks):
                                    continue
                                vo = float(np.mean([p[x][li][pi] for x in ks]))
                                vr = p[kr][li][pi]
                                d.append(vo - vr)
                                so.append(vo)
                                sr.append(vr)
                            if not d:
                                continue
                            d = np.array(d)
                            out[f"{kind}|{cond}|{l}|{k}|{tag}|{pname}"] = dict(
                                n=len(d), oracle=float(np.mean(so)), rand=float(np.mean(sr)), diff=float(d.mean()),
                                ci=boot(lambda i, d=d: float(d[i].mean()), len(d)), _per_item=d.tolist())
    return out


def curves(info, push):
    """Mean survival curves over items (layers l .. L-1) for plotting."""
    acc = {}
    for rec in push:
        for key, v in rec["persist"].items():
            acc.setdefault(key, []).append(np.asarray(v))
    return {k: np.round(np.mean(np.stack(v), 0), 4).tolist() for k, v in acc.items()}


# ---------------------------------------------------------------- main
ITEMS = json.load(open(os.path.join(HERE, "data", "items.json"), encoding="utf-8"))["d3"]
I8 = json.load(open(os.path.join(HERE, "data", "items_d8.json"), encoding="utf-8"))
A_SETS = dict(settle=ITEMS["A"], flip=I8["A"])
summary = dict(model=KEY, smoke=SMOKE, targets={})
for target in ("settle", "flip"):
    got = load(target)
    if got is None:
        print(f"{target}: no push results")
        continue
    info, clean, push, H, dirs = got
    L, layers = info["L"], info["layers"]
    sesoi = 1.0 / L if target == "settle" else 0.10
    binary = target == "flip"
    S = dict(info=info, sesoi=sesoi, n_push_items=len(push))
    S["readability"] = readability(info, clean, H, dirs, binary)
    cells = {}
    for cond in ("a", "b", "c"):
        for l in layers:
            for kind in (("o", "os") if cond == "c" else ("o",)):
                for k in K:
                    for s in (1, -1):
                        r = cell(push, cond, l, kind, k, s, sesoi)
                        if r is not None:  # §2: flag cells at layers where this oracle does not read the target
                            r["readable_layer"] = S["readability"][f"{kind}_{l}"]["verdict"] == "pass"
                            cells[f"{kind}|{cond}|{l}|{k}|{s}"] = r
    S["cells"] = cells
    S["moves"] = {f"{kind}|{cond}": moves(cells, cond, kind, layers)
                  for cond in ("a", "b", "c") for kind in (("o", "os") if cond == "c" else ("o",))}
    S["persistence"] = persistence(info, push)
    S["curves"] = curves(info, push)
    summary["targets"][target] = S

# ---------------------------------------------------------------- hypothesis verdicts (§5)
T = summary["targets"]
V = {}
if "flip" in T:
    V["positive_control"] = T["flip"]["moves"]["o|a"]["moves"]
if "settle" in T:
    mv = T["settle"]["moves"]
    a_ = mv["o|a"]["moves"]
    b_ = mv["o|b"]["moves"]
    c_ = mv["o|c"]["moves"] or mv["os|c"]["moves"]
    V["settle_moves"] = dict(a=a_, b=b_, c=c_, c_o=mv["o|c"]["moves"], c_os=mv["os|c"]["moves"])
    pc = V.get("positive_control")
    if not pc:
        V["verdict"] = "UNINFORMATIVE (positive control failed: flips do not move under condition a)"
    elif (b_ or c_) and not a_:
        V["verdict"] = "H-REDERIVE"
    elif not (a_ or b_ or c_):
        V["verdict"] = "H-NONLINEAR"
    elif a_:
        V["verdict"] = "NEITHER (settling depth moves from the reading position onward: a layer or strength effect, not position)"
    else:
        V["verdict"] = "MIXED"
    ell = T["settle"]["info"]["ell"]
    key = f"o|a|{ell}|2|plus{OFF}|P2"
    ps, pf = T["settle"]["persistence"].get(key), (T.get("flip", {}).get("persistence", {}).get(f"o|a|{T['flip']['info']['ell']}|2|plus{OFF}|P2") if "flip" in T else None)
    if ps:
        cls = "overwritten" if (ps["diff"] <= -0.25 and ps["ci"][1] is not None and ps["ci"][1] < 0) else \
              ("amplified" if (ps["diff"] >= 0.25 and ps["ci"][0] is not None and ps["ci"][0] > 0) else "kept")
        V["persistence_settle"] = dict(diff=ps["diff"], ci=ps["ci"], oracle=ps["oracle"], rand=ps["rand"], class_=cls)
        if pf:
            ds, df = np.array(ps["_per_item"]), np.array(pf["_per_item"])
            rng = np.random.default_rng(0)
            bb = [ds[rng.integers(0, len(ds), len(ds))].mean() - df[rng.integers(0, len(df), len(df))].mean() for _ in range(NB)]
            V["persistence_settle_minus_flip"] = dict(diff=float(ds.mean() - df.mean()),
                                                      ci=[float(np.quantile(bb, 0.025)), float(np.quantile(bb, 0.975))])
    # go / no-go for the 3-model study (§6)
    V["go_3_models"] = V["verdict"] in ("H-REDERIVE", "H-NONLINEAR")
summary["verdicts"] = V
for t in T.values():
    for v in t["persistence"].values():
        v.pop("_per_item", None)
path = os.path.join(RES, f"{KEY}{'_smoke' if SMOKE else ''}_summary.json")
json.dump(summary, open(path, "w"), indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else float(x))

# ---------------------------------------------------------------- printout
for target, S in T.items():
    print(f"\n== {target} ({KEY}) SESOI {S['sesoi']:.4f}, {S['n_push_items']} items; readability:",
          {k: (round(v['est'], 3) if v['est'] is not None else None, v['verdict']) for k, v in S["readability"].items() if isinstance(v, dict)})
    for key, r in S["cells"].items():
        if "effect" not in r:
            print(f"  {key:<18} not bracketed (n_excl {r['n_excluded']})")
            continue
        print(f"  {key:<18} kl~{r['kl_median']:7.2f} eff={r['effect']:+.4f} ci=[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
              f"norm={r['norm_matched']['effect'] if r['norm_matched']['effect'] is None else round(r['norm_matched']['effect'], 4)} "
              f"n={r['n']} excl={r['n_excluded']} {r['verdict']}{' *' if r['controllable'] else ''}{'' if r['readable_layer'] else ' (not readable here)'}")
    print("  moves:", {k: v["moves"] for k, v in S["moves"].items()})
print("\nVERDICTS", json.dumps(V, indent=1, default=str))
