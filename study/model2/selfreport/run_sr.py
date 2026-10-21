"""Paper 8 self-report arm (selfreport/PREDICTIONS.md).

python run_sr.py <stage> [n_recipients]

formatcheck  (pre-freeze, items outside the run) letter mass and argmax of the follow-up readout, both orders.
fitlens      tuned lens: per-layer affine map from block output l to the last block output, ridge
             regression on positions of texts outside the run (arithmetic items not in arm S2, ARC
             prompts), ridge strength chosen per layer on held-out positions by KL to the model's output.
selftest     engine checks on same-product groups outside the run.
kill         arm-S2 recipients: clean, donors (teacher-forced on the shared answer) and the cells (iv)@21,
             (iv)@14, (iii+)@21, one teacher-forced pass each over prompt + answer; settle layers of every
             digit under the logit lens and the tuned lens. res/kill.jsonl
sr           only if analyze_sr.py kill wrote verdict PASS: prompt + answer + follow-up turn, both label
             orders, cells (iv)@21, (i)@7, (ii-r)@21, (ii-r)@7, (ii)@7 x donors, plus Gaussian additive
             controls at the (iv)@21 positions; letter readout r, KL window, settle layers. res/sr.jsonl
With n_recipients, kill writes kill_smoke.jsonl (timing only); sr has no smoke mode (it would read run items).
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
P2 = os.path.join(HERE, "..", "patch2")
OLD = os.path.join(HERE, "..", "patch")
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from srlib import SRModel  # noqa: E402
from specs import D3  # noqa: E402
from p6lib import gaussian  # noqa: E402

STAGE = sys.argv[1]
N_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
KEY = "sm2-1.7b"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
T0 = time.time()
KILL_CELLS = [("iv", 18), ("iv", 12), ("iiip", 18)]
SR_CELLS = [("iv", 18), ("i", 6), ("iir", 18), ("iir", 6), ("ii", 6)]
GAUSS_R, GAUSS_C, GAUSS_L = 6, [0.125, 0.25, 0.5, 1, 2, 4, 8], 18


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = D3()
P = spec.P


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def g_item(i):
    a, b = (int(v) for v in i[2:].split("*"))
    return dict(id=i, set="G", a=a, b=b, op="*", answer=a * b, prompt=f"What is {a} * {b}?", prefill=f"{a} * {b} = ")


def layout(m, it):
    """As ../patch2/run_patch2.py: prompt ids, reading position, question-token positions."""
    text = m.chat(it["prompt"]) + it["prefill"]
    ids, start = spec.prompt_ids(m, it, "direct")
    enc = m.tok(text, add_special_tokens=not text.startswith(m.tok.bos_token or "\x00"), return_offsets_mapping=True)
    assert enc["input_ids"] == ids
    a = text.index(it["prompt"])
    b = a + len(it["prompt"])
    src = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < b and e > a]
    assert src and max(src) < start
    return ids, start, src


def pos_sets(n, start, src, K):
    ans = list(range(n, n + K - 1))  # positions holding answer tokens 1..K-1 (as patch2)
    return dict(i=list(src), ii=list(range(start, n)), iir=[start], iv=ans, iiip=list(range(n)) + ans)


def digit_pos(m, n, cont):
    _, idx = P.answer_span(m.tok, cont)
    return [n - 1 + k for k in idx]  # the position whose output predicts digit k


def settle_rows(T1):
    """T1 [B, ndig, L] -> per row list of settle layers (paper 4's settle_top1)."""
    return [[int(P.settle_top1(T1[b, k])) for k in range(T1.shape[1])] for b in range(T1.shape[0])]


def build_patch(rows, B, W, S, first, src_of):
    patch = {}
    for l in sorted({rw["l"] for rw in rows if rw.get("cond")}):
        mask = torch.zeros(B, W, dtype=torch.bool)
        src = torch.arange(B)
        for j, rw in enumerate(rows):
            if rw.get("cond") and rw["l"] == l:
                mask[first + j, S[rw["cond"]]] = True
                src[first + j] = src_of(rw)
        patch[l] = (mask, src)
    return patch


m = SRModel(KEY)
L, d = m.L, m.d
POOL = os.path.join(OLD, "res", "pool_clean.jsonl")
pool = {r["id"]: r for r in read_jsonl(POOL)}
pairs = json.load(open(os.path.join(P2, "res", "pairs2.json"), encoding="utf-8"))["S"]
RUN_ITEMS = set()
for x in pairs:
    RUN_ITEMS |= {x["r"], *x["far"], *x["null"]}
LA, LB = m.letter_ids()
LENS_PATH = os.path.join(RES, "tuned_lens.pt")


def load_lens():
    m.tuned = torch.load(LENS_PATH, weights_only=False)["W"].to(m.dev)


def outside_groups(k, seed):
    """Same-product groups of correct 4-digit G items outside the run (as patch2's selftest)."""
    byc = {}
    for i, r in sorted(pool.items()):
        if r["set"] == "G" and r["correct"] == 1 and i not in RUN_ITEMS and len(r["cont"]) == 4:
            byc.setdefault(tuple(r["cont"]), []).append(i)
    groups = [g for g in byc.values() if len(g) >= 2]
    rng = np.random.default_rng(seed)
    return [groups[j] for j in rng.choice(len(groups), k, replace=False)]


# ============================================================= formatcheck
if STAGE == "formatcheck":
    stats = []
    for grp in outside_groups(10, 11):
        it = g_item(grp[0])
        ids, _, _ = layout(m, it)
        c = pool[grp[0]]["cont"]
        for order in (0, 1):
            seq = ids + c + m.tail_ids(order)
            h, _ = m.tf_pass([seq])
            lp = torch.log_softmax(m.hf.lm_head(h[:, -1]).float(), -1)[0]
            pa, pb = float(lp[LA].exp()), float(lp[LB].exp())
            top = m.tok.decode([int(lp.argmax())])
            stats.append(dict(order=order, mass=pa + pb, top=top, r=(1 if order == 0 else -1) * float(lp[LA] - lp[LB])))
    if True:
        it = g_item(outside_groups(1, 12)[0][0])
        ids, _, _ = layout(m, it)
        log("tail tokens", len(m.tail_ids(0)), len(m.tail_ids(1)), "| decoded tail round-trips:",
            m.tok.decode(m.tail_ids(0)) == __import__("srlib").tail_text(0))
    mass = [s["mass"] for s in stats]
    log(f"formatcheck on {len(stats)} rows outside the run: letter mass median {np.median(mass):.3f} min {min(mass):.3f}; "
        f"argmax {sorted({s['top'] for s in stats})}; mean r (step-by-step minus just-knew) order0 "
        f"{np.mean([s['r'] for s in stats if s['order'] == 0]):+.2f} order1 {np.mean([s['r'] for s in stats if s['order'] == 1]):+.2f}")
    sys.exit(0)

# ================================================================ fitlens
if STAGE == "fitlens":
    from spec_d8 import PREFILL as D8P, user_text
    rng = np.random.default_rng(5)
    arith = [i for i, r in sorted(pool.items()) if r["set"] == "G" and i not in RUN_ITEMS and r["correct"] == 1]
    arith = [arith[j] for j in rng.choice(len(arith), 1500, replace=False)]
    seqs = []
    for i in arith:
        ids, _, _ = layout(m, g_item(i))
        seqs.append(ids + pool[i]["cont"] + [m.tok.convert_tokens_to_ids("<|im_end|>")])
    arc = json.load(open(os.path.join(HERE, "..", "code", "data", "items_d8.json"), encoding="utf-8"))["A"][:500]
    for it in arc:
        seqs.append(m.ids(m.chat(user_text(it, False)) + D8P + it["answer"] + ")"))
    order = rng.permutation(len(seqs))
    hold = set(order[: len(seqs) // 10].tolist())
    XtX = torch.zeros(L - 1, d + 1, d + 1, dtype=torch.float64, device=m.dev)
    XtY = torch.zeros(L - 1, d + 1, d, dtype=torch.float64, device=m.dev)
    Hh = {l: [] for l in range(L)}
    npos = 0
    for k, si in enumerate(order.tolist()):
        _, cap = m.tf_pass([seqs[si]])
        Y = cap[L - 1][0, 1:].double()  # position 0 (attention sink) excluded
        if si in hold:
            sel = torch.arange(Y.shape[0])
            for l in range(L):
                Hh[l].append(cap[l][0, 1:][sel].cpu())
            continue
        npos += Y.shape[0]
        for l in range(L - 1):
            X = torch.cat([cap[l][0, 1:].double(), torch.ones(Y.shape[0], 1, dtype=torch.float64, device=m.dev)], 1)
            XtX[l] += X.T @ X
            XtY[l] += X.T @ Y
        if k % 200 == 0:
            log(f"fitlens {k}/{len(seqs)} positions {npos}")
    Hh = {l: torch.cat(v, 0) for l, v in Hh.items()}
    nh = Hh[0].shape[0]
    sub = torch.randperm(nh, generator=torch.Generator().manual_seed(0))[:3000]
    ref = torch.log_softmax(m.hf.lm_head(m.norm(Hh[L - 1][sub].to(m.dev))).float(), -1)
    Wall = torch.zeros(L - 1, d + 1, d, dtype=torch.float32)
    report = []
    for l in range(L - 1):
        A = XtX[l]
        scale = float(torch.diagonal(A)[:-1].mean())
        Xh = torch.cat([Hh[l][sub].to(m.dev).double(), torch.ones(len(sub), 1, dtype=torch.float64, device=m.dev)], 1)
        best = None
        for lam in (1e-4, 1e-3, 1e-2, 1e-1):
            Wl = torch.linalg.solve(A + lam * scale * torch.eye(d + 1, dtype=torch.float64, device=m.dev), XtY[l])
            pred = (Xh @ Wl).to(torch.bfloat16)
            lp = torch.log_softmax(m.hf.lm_head(m.norm(pred)).float(), -1)
            kl = float((ref.exp() * (ref - lp)).sum(-1).mean())
            if best is None or kl < best[0]:
                best = (kl, lam, Wl)
        lp0 = torch.log_softmax(m.hf.lm_head(m.norm(Hh[l][sub].to(m.dev))).float(), -1)
        kl0 = float((ref.exp() * (ref - lp0)).sum(-1).mean())
        Wall[l] = best[2].float().cpu()
        report.append(dict(layer=l, kl_tuned=best[0], lam=best[1], kl_logit=kl0))
        log(f"layer {l:2d}: held-out KL tuned {best[0]:.3f} (lambda {best[1]:g}) vs logit lens {kl0:.3f}")
    torch.save(dict(W=Wall, report=report, n_train_positions=npos, n_heldout_positions=nh), LENS_PATH)
    json.dump(dict(report=report, n_train_positions=npos, n_heldout_positions=nh, n_seqs=len(seqs)),
              open(os.path.join(RES, "tuned_lens_report.json"), "w"), indent=1)
    log("DONE")
    sys.exit(0)

load_lens()

# =============================================================== selftest
if STAGE == "selftest":
    worst = dict(clean_settle_match=0, clean_rows=0, write=0.0, causal=0.0, gauss_cos=1.0, tuned_final=0, follow_causal=0.0)
    gdir = torch.tensor(gaussian(1, d, seed=99)[0], dtype=torch.float32)
    for grp in outside_groups(3, 0):
        rit, dit = g_item(grp[0]), g_item(grp[1])
        rid, start, src = layout(m, rit)
        did, st_d, src_d = layout(m, dit)
        n = len(rid)
        assert len(did) == n and st_d == start and src_d == src
        c = pool[grp[0]]["cont"]
        K = len(c)
        S = pos_sets(n, start, src, K)
        dp = digit_pos(m, n, c)
        tail = m.tail_ids(0)
        rows = [dict(cond=cc, l=l) for cc, l in SR_CELLS + KILL_CELLS] + [dict(g=1)]
        B = 2 + len(rows)
        seqs = [rid + c + tail, did + c + tail] + [rid + c + tail] * len(rows)
        W = n + K - 1
        patch = build_patch(rows, B, W, S, 2, lambda rw: 1)
        add = {GAUSS_L: torch.zeros(B, d)}
        pm = torch.zeros(B, len(seqs[0]))
        add[GAUSS_L][B - 1] = 50.0 * gdir
        pm[B - 1, S["iv"]] = 1
        h, cap = m.tf_pass(seqs, patch, add, pm)
        st = settle_rows(m.lens_top1(cap, dp).cpu())
        worst["clean_rows"] += 1
        worst["clean_settle_match"] += int(st[0] == pool[grp[0]]["settle"])
        tl = m.lens_top1(cap, dp, tuned=True).cpu()
        ll = m.lens_top1(cap, dp).cpu()
        worst["tuned_final"] += int((tl[..., -1] != ll[..., -1]).sum())
        for j, rw in enumerate(rows[:-1]):
            b, l, ps = 2 + j, rw["l"], S[rw["cond"]]
            worst["write"] = max(worst["write"], float((cap[l][b, ps].float() - cap[l][1, ps].float()).abs().max()))
            first = min(ps)
            if first > 0:
                worst["causal"] = max(worst["causal"], max(float((cap[k][b, :first].float() - cap[k][0, :first].float()).abs().max())
                                                           for k in range(L)))
        dg = (cap[GAUSS_L][B - 1, S["iv"]].float() - cap[GAUSS_L][0, S["iv"]].float())
        worst["gauss_cos"] = min(worst["gauss_cos"], float(torch.nn.functional.cosine_similarity(dg, (50.0 * gdir).to(m.dev)[None], dim=-1).min()))
        # the follow-up turn does not change anything before it: two rows of equal length in one batch that
        # differ only in the follow-up text (label order 0 vs 1) are identical at every earlier position
        t0, t1 = m.tail_ids(0), m.tail_ids(1)
        assert len(t0) == len(t1)
        fd = next(k for k in range(len(t0)) if t0[k] != t1[k])
        _, cap2 = m.tf_pass([rid + c + t0, rid + c + t1])
        worst["follow_causal"] = max(worst["follow_causal"], max(float((cap2[k][0, :n + K + fd].float() - cap2[k][1, :n + K + fd].float()).abs().max())
                                                                 for k in range(L)))
        log(f"{grp[0]} <- {grp[1]}: {worst}")
    log("selftest", worst)
    assert worst["write"] == 0.0, "the patch did not write exactly the donor row's state"
    assert worst["causal"] == 0.0, "a patch changed earlier positions"
    assert worst["gauss_cos"] > 0.99, "the Gaussian add did not land as specified"
    assert worst["tuned_final"] == 0, "tuned lens at the last layer is not the model's output"
    assert worst["clean_settle_match"] >= worst["clean_rows"] - 1, "teacher-forced settle differs from the pool"
    assert worst["follow_causal"] == 0.0, "the follow-up changed earlier positions"
    log("SELFTEST PASSED")
    sys.exit(0)


def donors_of(pr):
    return [(dd, "far") for dd in pr["far"]] + [(dd, "null") for dd in pr["null"]]


# =================================================================== kill
if STAGE == "kill":
    path = os.path.join(RES, "kill" + ("_smoke" if N_LIMIT else "") + ".jsonl")
    done = {r["r"] for r in read_jsonl(path)}
    recs = pairs[:N_LIMIT] if N_LIMIT else pairs
    todo = [x for x in recs if x["r"] not in done]
    log(f"kill: {len(todo)} of {len(recs)} recipients")
    with open(path, "a", encoding="utf-8") as f:
        for nn, pr in enumerate(todo):
            rit = g_item(pr["r"])
            rid, start, src = layout(m, rit)
            n = len(rid)
            c = pool[pr["r"]]["cont"]
            K = len(c)
            S = pos_sets(n, start, src, K)
            dp = digit_pos(m, n, c)
            dons = donors_of(pr)
            dids = []
            for dd, _ in dons:
                ids_d, st_d, src_d = layout(m, g_item(dd))
                assert len(ids_d) == n and st_d == start and src_d == src and pool[dd]["cont"] == c
                dids.append(ids_d)
            nd = len(dids)
            rows = [dict(donor=dd, role=role, cond=cc, l=l, di=k) for k, (dd, role) in enumerate(dons) for cc, l in KILL_CELLS]
            B = 1 + nd + len(rows)
            seqs = [rid + c] + [x + c for x in dids] + [rid + c] * len(rows)
            h, cap = m.tf_pass(seqs, build_patch(rows, B, n + K - 1, S, 1 + nd, lambda rw: 1 + rw["di"]))
            ll, tl = m.lens_top1(cap, dp).cpu(), m.lens_top1(cap, dp, tuned=True).cpu()
            sl, stl = settle_rows(ll), settle_rows(tl)
            ans = ll[:, :, -1].tolist()
            del cap, h
            rec = dict(r=pr["r"], s=pr["s"], K=K, clean=dict(ll=sl[0], tl=stl[0], ans=ans[0]),
                       don={dd: dict(ll=sl[1 + k], tl=stl[1 + k], role=role, same=int(ans[1 + k] == ans[0])) for k, (dd, role) in enumerate(dons)},
                       rows=[dict(rw, ll=sl[1 + nd + j], tl=stl[1 + nd + j], same=int(ans[1 + nd + j] == ans[0])) for j, rw in enumerate(rows)])
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if nn % 20 == 0:
                log(f"kill {nn}/{len(todo)} {pr['r']} clean ll {sl[0]} tl {stl[0]}")
    log("DONE")
    sys.exit(0)

# ===================================================================== sr
assert STAGE == "sr", STAGE
ks = json.load(open(os.path.join(RES, "kill_summary.json"), encoding="utf-8"))
assert ks["verdict"] == "PASS", f"kill check verdict is {ks['verdict']}: step 3 is not run (PREDICTIONS.md)"
gauss = torch.tensor(gaussian(GAUSS_R, d, seed=1234), dtype=torch.float32)
path = os.path.join(RES, "sr" + ("_smoke" if N_LIMIT else "") + ".jsonl")
done = {(r["r"], r["order"]) for r in read_jsonl(path)}
recs = pairs[:N_LIMIT] if N_LIMIT else pairs
log(f"sr: {len(recs)} recipients x 2 orders; {len(done)} done")
with open(path, "a", encoding="utf-8") as f:
    for nn, pr in enumerate(recs):
        t_it = time.time()
        rit = g_item(pr["r"])
        rid, start, src = layout(m, rit)
        n = len(rid)
        c = pool[pr["r"]]["cont"]
        K = len(c)
        S = pos_sets(n, start, src, K)
        dp = digit_pos(m, n, c)
        dons = donors_of(pr)
        dids = [layout(m, g_item(dd))[0] for dd, _ in dons]
        nd = len(dids)
        # Gaussian scale: median distance between far donors and the recipient at the (iv) positions, layer 21
        _, cap0 = m.tf_pass([rid + c] + [x + c for x in dids])
        far_k = [1 + k for k, (_, role) in enumerate(dons) if role == "far"]
        sigma = float(torch.median(torch.stack([(cap0[GAUSS_L][k, S["iv"]].float() - cap0[GAUSS_L][0, S["iv"]].float()).norm(dim=-1)
                                                for k in far_k]).flatten()))
        del cap0
        for order in (0, 1):
            if (pr["r"], order) in done:
                continue
            tail = m.tail_ids(order)
            sgn = 1.0 if order == 0 else -1.0
            rows = [dict(donor=dd, role=role, cond=cc, l=l, di=k) for k, (dd, role) in enumerate(dons) for cc, l in SR_CELLS]
            rows += [dict(gj=j, gc=cc) for j in range(GAUSS_R) for cc in GAUSS_C]
            out_rows, clean, don = [], None, {}
            step = MAX_ROWS - 1 - nd
            for c0 in range(0, len(rows), step):
                ch = rows[c0:c0 + step]
                B = 1 + nd + len(ch)
                seqs = [rid + c + tail] + [x + c + tail for x in dids] + [rid + c + tail] * len(ch)
                T = len(seqs[0])
                patch = build_patch(ch, B, n + K - 1, S, 1 + nd, lambda rw: 1 + rw["di"])
                add = {GAUSS_L: torch.zeros(B, d)}
                pm = torch.zeros(B, T)
                for j, rw in enumerate(ch):
                    if "gj" in rw:
                        add[GAUSS_L][1 + nd + j] = rw["gc"] * sigma * gauss[rw["gj"]]
                        pm[1 + nd + j, S["iv"]] = 1
                h, cap = m.tf_pass(seqs, patch, add, pm)
                sl = settle_rows(m.lens_top1(cap, dp).cpu())
                stl = settle_rows(m.lens_top1(cap, dp, tuned=True).cpu())
                del cap
                lg = m.hf.lm_head(h[:, T - 1]).float()
                lpl = torch.log_softmax(lg, -1)
                r = (sgn * (lg[:, LA] - lg[:, LB])).tolist()
                pab = (lpl[:, LA].exp() + lpl[:, LB].exp()).tolist()
                kl = torch.zeros(B, device=m.dev)
                for p0 in range(n, T, 8):  # KL(clean row || row) over the window n .. T-1 (outputs from the first answer position on)
                    lp = m.window_logp(h, p0, min(T, p0 + 8))
                    kl += (lp[0:1].exp() * (lp[0:1] - lp)).sum(-1).clamp_min(0).sum(-1)
                    del lp
                kl = kl.tolist()
                del h
                if clean is None:
                    clean = dict(r=r[0], pab=pab[0], ll=sl[0], tl=stl[0])
                for k, (dd, role) in enumerate(dons):
                    don.setdefault(dd, dict(r=r[1 + k], pab=pab[1 + k], ll=sl[1 + k], tl=stl[1 + k], role=role))
                for j, rw in enumerate(ch):
                    b = 1 + nd + j
                    out_rows.append(dict(rw, r=r[b], pab=pab[b], kl=kl[b], ll=sl[b], tl=stl[b]))
            f.write(json.dumps(dict(r=pr["r"], s=pr["s"], K=K, order=order, sigma=sigma, clean=clean, don=don, rows=out_rows)) + "\n")
            f.flush()
        if nn % 10 == 0:
            log(f"sr {nn}/{len(recs)} {pr['r']} {time.time() - t_it:.1f}s sigma {sigma:.1f}")
log("DONE")
