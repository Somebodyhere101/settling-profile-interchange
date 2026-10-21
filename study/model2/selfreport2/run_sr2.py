"""Paper 8 self-report arm 2 (selfreport2/PREDICTIONS.md): per-digit report under temporally isolated patches.

python run_sr2.py <formatcheck|selftest|sr2>

Engine: ../selfreport/srlib.SRModel (unchanged): one teacher-forced pass per batch over prompt + shared
answer + report turn; in-batch interchange at fixed absolute positions of one block; Gaussian additive rows.

Report turns ("tails"), each appended after the answer as the model's own generation left it:
  digit question (primary), 4 cyclic option rotations: option letter j names digit (j + k) mod 4;
  A/B question (secondary), 2 orders (as ../selfreport).
Cells per donor: (iv)@21 answer positions only (temporally isolated: no report position is patched);
(iv+rep)@21 answer AND report positions (direct-steering control); (i)@7; (ii-r)@21; (ii)@7.
Gaussian additive control at the (iv) positions, block 21.
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SR1 = os.path.join(HERE, "..", "selfreport")
P2 = os.path.join(HERE, "..", "patch2")
OLD = os.path.join(HERE, "..", "patch")
sys.path.insert(0, SR1)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from srlib import SRModel  # noqa: E402
from specs import D3  # noqa: E402
from p6lib import gaussian  # noqa: E402

STAGE = sys.argv[1]
KEY = "sm2-1.7b"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
T0 = time.time()
CELLS = [("iv", 18), ("ivrep", 18), ("i", 6), ("iir", 18), ("ii", 6)]
GAUSS_R, GAUSS_C, GAUSS_L = 4, [0.125, 0.25, 0.5, 1, 2, 4, 8], 18
ORD = ["first", "second", "third", "fourth"]
DQ = "Which digit of your answer were you least sure of: the first, second, third or fourth? Answer A, B, C or D."
ABQ = "Did you work that out step by step, or did you just know it? Answer A or B."
ABO = ("I worked it out step by step.", "I just knew it.")
PREFILL = "The answer is ("
HEAD = "<|im_end|>\n<|im_start|>user\n"
FOOT = "<|im_end|>\n<|im_start|>assistant\n" + PREFILL


def tail_text(kind, k):
    if kind == "digit":  # rotation k: letter j names digit (j + k) % 4
        opts = "\n".join(f"{'ABCD'[j]}) the {ORD[(j + k) % 4]} digit" for j in range(4))
        return HEAD + DQ + "\n\n" + opts + FOOT
    a, b = ABO if k == 0 else ABO[::-1]
    return HEAD + f"{ABQ}\n\nA) {a}\nB) {b}" + FOOT


TAILS = [("digit", k) for k in range(4)] + [("ab", 0), ("ab", 1)]


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
    text = m.chat(it["prompt"]) + it["prefill"]
    ids, start = spec.prompt_ids(m, it, "direct")
    enc = m.tok(text, add_special_tokens=not text.startswith(m.tok.bos_token or "\x00"), return_offsets_mapping=True)
    assert enc["input_ids"] == ids
    a = text.index(it["prompt"])
    b = a + len(it["prompt"])
    src = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < b and e > a]
    assert src and max(src) < start
    return ids, start, src


def pos_sets(n, start, src, K, T):
    ans = list(range(n, n + K - 1))  # as patch2 / selfreport
    rep = list(range(n + K - 1, T))  # the last answer token's position and every report-turn position
    return dict(i=list(src), ii=list(range(start, n)), iir=[start], iv=ans, ivrep=ans + rep)


def digit_pos(m, n, cont):
    _, idx = P.answer_span(m.tok, cont)
    return [n - 1 + k for k in idx]


def settle_rows(T1):
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
pool = {r["id"]: r for r in read_jsonl(os.path.join(OLD, "res", "pool_clean.jsonl"))}
pairs = json.load(open(os.path.join(P2, "res", "pairs2.json"), encoding="utf-8"))["S"]
RUN_ITEMS = set()
for x in pairs:
    RUN_ITEMS |= {x["r"], *x["far"], *x["null"]}
LET = [m.tok.convert_tokens_to_ids(c) for c in "ABCD"]
TAIL_IDS = {t: m.tok(tail_text(*t), add_special_tokens=False)["input_ids"] for t in TAILS}
assert len({len(TAIL_IDS[t]) for t in TAILS if t[0] == "digit"}) == 1, "digit rotations differ in length"


def outside_groups(k, seed):
    byc = {}
    for i, r in sorted(pool.items()):
        if r["set"] == "G" and r["correct"] == 1 and i not in RUN_ITEMS and len(r["cont"]) == 4:
            byc.setdefault(tuple(r["cont"]), []).append(i)
    groups = [g for g in byc.values() if len(g) >= 2]
    rng = np.random.default_rng(seed)
    return [groups[j] for j in rng.choice(len(groups), k, replace=False)]


def letter_logits(h, T, kind):
    lg = m.hf.lm_head(h[:, T - 1]).float()
    lp = torch.log_softmax(lg, -1)
    nl = 4 if kind == "digit" else 2
    return lg[:, LET[:nl]].tolist(), lp[:, LET[:nl]].exp().sum(-1).tolist()


# ============================================================ formatcheck
if STAGE == "formatcheck":
    stats = {t: [] for t in TAILS}
    for grp in outside_groups(10, 11):
        it = g_item(grp[0])
        ids, _, _ = layout(m, it)
        c = pool[grp[0]]["cont"]
        for t in TAILS:
            h, _ = m.tf_pass([ids + c + TAIL_IDS[t]])
            lg, mass = letter_logits(h, len(ids + c + TAIL_IDS[t]), t[0])
            top = int(np.argmax(lg[0]))
            stats[t].append(dict(mass=mass[0], top=top, digit=(top + t[1]) % 4 if t[0] == "digit" else None))
    for t, v in stats.items():
        log(t, f"letter mass median {np.median([x['mass'] for x in v]):.3f} min {min(x['mass'] for x in v):.3f}; argmax letters "
               f"{dict(zip(*np.unique([x['top'] for x in v], return_counts=True)))}" +
            (f"; argmax digits {dict(zip(*np.unique([x['digit'] for x in v], return_counts=True)))}" if t[0] == "digit" else ""))
    sys.exit(0)

# =============================================================== selftest
if STAGE == "selftest":
    worst = dict(write=0.0, causal=0.0, rep_iso=0.0, rep_reads=float("inf"), ivrep_rep_write=0.0, gauss_cos=1.0)
    gdir = torch.tensor(gaussian(1, d, seed=99)[0], dtype=torch.float32)
    for grp in outside_groups(3, 0):
        rid, start, src = layout(m, g_item(grp[0]))
        did, _, _ = layout(m, g_item(grp[1]))
        n = len(rid)
        c = pool[grp[0]]["cont"]
        K = len(c)
        tail = TAIL_IDS[("digit", 1)]
        T = n + K + len(tail)
        S = pos_sets(n, start, src, K, T)
        rows = [dict(cond=cc, l=l) for cc, l in CELLS] + [dict(g=1)]
        B = 2 + len(rows)
        seqs = [rid + c + tail, did + c + tail] + [rid + c + tail] * len(rows)
        add = {GAUSS_L: torch.zeros(B, d)}
        pm = torch.zeros(B, T)
        add[GAUSS_L][B - 1] = 50.0 * gdir
        pm[B - 1, S["iv"]] = 1
        h, cap = m.tf_pass(seqs, build_patch(rows, B, T, S, 2, lambda rw: 1), add, pm)
        for j, rw in enumerate(rows[:-1]):
            b, l, ps = 2 + j, rw["l"], S[rw["cond"]]
            worst["write"] = max(worst["write"], float((cap[l][b, ps].float() - cap[l][1, ps].float()).abs().max()))
            first = min(ps)
            worst["causal"] = max(worst["causal"], max(float((cap[k][b, :first].float() - cap[k][0, :first].float()).abs().max()) for k in range(L)))
        bi = 2 + [r.get("cond") for r in rows].index("iv")
        # temporal isolation: under (iv)@21 the report positions are not patched; through block 21 they equal the
        # clean recipient's exactly, and only from block 22 on do they read the patched answer states (attention)
        rep = S["ivrep"][K - 1:]
        worst["rep_iso"] = max(worst["rep_iso"], max(float((cap[k][bi, rep].float() - cap[k][0, rep].float()).abs().max()) for k in range(19)))
        worst["rep_reads"] = min(worst["rep_reads"], float((cap[L - 1][bi, rep].float() - cap[L - 1][0, rep].float()).abs().max()))
        # ... while under (iv+rep)@21 they are exactly the donor's
        br = 2 + [r.get("cond") for r in rows].index("ivrep")
        worst["ivrep_rep_write"] = max(worst["ivrep_rep_write"], float((cap[18][br, rep].float() - cap[18][1, rep].float()).abs().max()))
        dg = cap[GAUSS_L][B - 1, S["iv"]].float() - cap[GAUSS_L][0, S["iv"]].float()
        worst["gauss_cos"] = min(worst["gauss_cos"], float(torch.nn.functional.cosine_similarity(dg, (50.0 * gdir).to(m.dev)[None], dim=-1).min()))
        log(grp, worst)
    log("selftest", worst)
    assert worst["write"] == 0.0 and worst["causal"] == 0.0 and worst["ivrep_rep_write"] == 0.0
    assert worst["rep_iso"] == 0.0, "(iv)@21 changed the report positions at or below block 21"
    assert worst["rep_reads"] > 0.0, "(iv)@21 does not reach the report positions at all"
    assert worst["gauss_cos"] > 0.99
    log("SELFTEST PASSED")
    sys.exit(0)

# ==================================================================== sr2
assert STAGE == "sr2", STAGE
gauss = torch.tensor(gaussian(GAUSS_R, d, seed=1234), dtype=torch.float32)
path = os.path.join(RES, "sr2.jsonl")
done = {(r["r"], r["kind"], r["k"]) for r in read_jsonl(path)}
log(f"sr2: {len(pairs)} recipients x {len(TAILS)} report turns; {len(done)} done")
with open(path, "a", encoding="utf-8") as f:
    for nn, pr in enumerate(pairs):
        t_it = time.time()
        rid, start, src = layout(m, g_item(pr["r"]))
        n = len(rid)
        c = pool[pr["r"]]["cont"]
        K = len(c)
        dp = digit_pos(m, n, c)
        dons = [(x, "far") for x in pr["far"]] + [(x, "null") for x in pr["null"]]
        dids = [layout(m, g_item(x))[0] for x, _ in dons]
        nd = len(dids)
        _, cap0 = m.tf_pass([rid + c] + [x + c for x in dids])
        far_k = [1 + k for k, (_, role) in enumerate(dons) if role == "far"]
        S0 = pos_sets(n, start, src, K, n + K)
        sigma = float(torch.median(torch.stack([(cap0[GAUSS_L][k, S0["iv"]].float() - cap0[GAUSS_L][0, S0["iv"]].float()).norm(dim=-1)
                                                for k in far_k]).flatten()))
        del cap0
        for t in TAILS:
            if (pr["r"], t[0], t[1]) in done:
                continue
            tail = TAIL_IDS[t]
            T = n + K + len(tail)
            S = pos_sets(n, start, src, K, T)
            rows = [dict(donor=x, role=role, cond=cc, l=l, di=k) for k, (x, role) in enumerate(dons) for cc, l in CELLS]
            rows += [dict(gj=j, gc=cc) for j in range(GAUSS_R) for cc in GAUSS_C]
            out_rows, clean, don = [], None, {}
            step = MAX_ROWS - 1 - nd
            for c0 in range(0, len(rows), step):
                ch = rows[c0:c0 + step]
                B = 1 + nd + len(ch)
                seqs = [rid + c + tail] + [x + c + tail for x in dids] + [rid + c + tail] * len(ch)
                add = {GAUSS_L: torch.zeros(B, d)}
                pm = torch.zeros(B, T)
                for j, rw in enumerate(ch):
                    if "gj" in rw:
                        add[GAUSS_L][1 + nd + j] = rw["gc"] * sigma * gauss[rw["gj"]]
                        pm[1 + nd + j, S["iv"]] = 1
                h, cap = m.tf_pass(seqs, build_patch(ch, B, T, S, 1 + nd, lambda rw: 1 + rw["di"]), add, pm)
                sl = settle_rows(m.lens_top1(cap, dp).cpu())
                ans = m.lens_top1(cap, dp)[:, :, -1].cpu().tolist()
                del cap
                lg, mass = letter_logits(h, T, t[0])
                kl = torch.zeros(B, device=m.dev)
                for p0 in range(n, T, 8):
                    lp = m.window_logp(h, p0, min(T, p0 + 8))
                    kl += (lp[0:1].exp() * (lp[0:1] - lp)).sum(-1).clamp_min(0).sum(-1)
                    del lp
                kl = kl.tolist()
                del h
                if clean is None:
                    clean = dict(lg=lg[0], mass=mass[0], ll=sl[0], ans=ans[0])
                for k, (x, role) in enumerate(dons):
                    don.setdefault(x, dict(lg=lg[1 + k], mass=mass[1 + k], ll=sl[1 + k], role=role, same=int(ans[1 + k] == ans[0])))
                for j, rw in enumerate(ch):
                    b = 1 + nd + j
                    out_rows.append(dict(rw, lg=lg[b], mass=mass[b], kl=kl[b], ll=sl[b], same=int(ans[b] == ans[0])))
            f.write(json.dumps(dict(r=pr["r"], s=pr["s"], K=K, kind=t[0], k=t[1], sigma=sigma, clean=clean, don=don, rows=out_rows)) + "\n")
            f.flush()
        if nn % 10 == 0:
            log(f"sr2 {nn}/{len(pairs)} {pr['r']} {time.time() - t_it:.1f}s")
log("DONE")
