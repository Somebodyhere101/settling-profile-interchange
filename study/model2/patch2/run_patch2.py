"""Paper 8 patching arm 2 (patch2/PREDICTIONS.md): is settling depth held at the answer positions?

python run_patch2.py <stage> [n_recipients]

selftest  engine checks on same-product pairs that are not in the run (no far-null contrast is computed):
          the unpatched engine reproduces PatchModel.gen; forcing a row's own greedy answer changes
          nothing; a self-patch at every prompt and answer position is the identity; a patch writes
          exactly the source row's state at the patched positions and leaves every earlier position
          unchanged; with every position patched at layer l, the lens at layers >= l equals the donor
          row's and D equals the splice of the two clean lens tables.
pair2     arm S pairs of ../patch/res/pairs.json, keeping only donors whose clean answer tokens equal the
          recipient's (so answer positions align); recipients need >= 1 far and >= 1 null donor left.
          Writes res/pairs2.json; prints counts only.
patchS2   for each recipient, every batch holds: row 0 the clean recipient; then its donors, teacher-forced
          on the shared answer (the in-batch sources of the patch); then the patched rows, every donor x
          condition x layer. Conditions: (i) question tokens, (ii) reading position onward (prompt), (iii) all prompt
          positions, (iv) answer positions only, (ii+) reading position onward plus answer positions,
          (iii+) all prompt and answer positions. Layers 7/14/21/24.
With n_recipients, patchS2 writes res/patchS2_smoke.jsonl (timing only).
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
OLD = os.path.join(HERE, "..", "patch")
sys.path.insert(0, CODE)

from patchlib2 import PatchModel2  # noqa: E402
from specs import D3  # noqa: E402

STAGE = sys.argv[1]
N_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
KEY = "sm2-1.7b"
LAYERS = [6, 12, 18, 21]
CONDS = ["i", "ii", "iii", "iv", "iip", "iiip"]  # iip = (ii+), iiip = (iii+)
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
N_NEW = D3.n_new_mode["direct"]
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
T0 = time.time()


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
    """As ../patch/run_patch.py: prompt ids, reading position, question-token positions."""
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
    """K = number of shared answer tokens; the answer positions n .. n+K-2 hold answer tokens 1..K-1 and
    are the positions whose outputs predict answer tokens 2..K."""
    ans = list(range(n, n + K - 1))
    return dict(i=list(src), ii=list(range(start, n)), iii=list(range(n)), iv=ans,
                iip=list(range(start, n)) + ans, iiip=list(range(n)) + ans)


def outcome(m, it, toks, top1, keep_top1=False):
    oc = spec.outcome(m, it, "direct", toks, None, top1)
    tt = m.trunc(toks)
    ans, idx = P.answer_span(m.tok, tt)
    st = [P.settle_top1(top1[i]) for i in idx]
    oc["D_gen"] = float(np.mean(st[1:])) / m.L if len(st) > 1 else None
    oc["settle"] = st
    oc["cont"] = [int(t) for t in tt[:idx[-1] + 1]] if idx else []
    if keep_top1:
        oc["top1"] = [[int(v) for v in top1[i]] for i in idx]  # lens top-1 at each digit's step
    oc.pop("text", None)
    return oc


def build_patch(rows, B, W, S, first, src_of):
    """In-batch patch: rows[j] is batch row first + j, copying from batch row src_of(rows[j])."""
    patch = {}
    for l in LAYERS:
        mask = torch.zeros(B, W, dtype=torch.bool)
        src = torch.arange(B)
        for j, rw in enumerate(rows):
            if rw["l"] == l:
                mask[first + j, S[rw["cond"]]] = True
                src[first + j] = src_of(rw)
        if mask.any():
            patch[l] = (mask, src)
    return patch


def splice(r_top1, d_top1, l):
    """Settle layers of a row whose lens is the recipient's below l and the donor's from l up."""
    return [P.settle_top1(torch.tensor(a[:l] + b[l:])) for a, b in zip(r_top1, d_top1)]


m = PatchModel2(KEY)
L = m.L
POOL = os.path.join(OLD, "res", "pool_clean.jsonl")

# ================================================================= selftest
if STAGE == "selftest":
    pool = {r["id"]: r for r in read_jsonl(POOL)}
    used = set()
    for x in json.load(open(os.path.join(OLD, "res", "pairs.json"), encoding="utf-8"))["S"]:
        used |= {x["r"], *x["far"], *x["null"]}
    byc = {}
    for i, r in sorted(pool.items()):
        if r["set"] == "G" and r["correct"] == 1 and i not in used and len(r["cont"]) == 4:
            byc.setdefault(tuple(r["cont"]), []).append(i)
    groups = [g for g in byc.values() if len(g) >= 2]
    rng = np.random.default_rng(0)
    pick = [groups[j] for j in rng.choice(len(groups), 3, replace=False)]
    worst = dict(gen_tok=0, gen_top1=0, force_top1=0, self_tok=0, self_top1=0, write=0.0, causal=0.0, causal_top1=0,
                 splice_top1=0.0, splice_D_match=0, splice_D_rows=0)
    for grp in pick:
        rit, dit = g_item(grp[0]), g_item(grp[1])
        rid, start, src = layout(m, rit)
        did, st_d, src_d = layout(m, dit)
        n = len(rid)
        assert len(did) == n and st_d == start and src_d == src
        c = pool[grp[0]]["cont"]
        K = len(c)
        W = n + K - 1
        S = pos_sets(n, start, src, K)
        # 1. the unpatched engine reproduces PatchModel.gen (mixed prompts); forcing the clean answer is a no-op
        g0 = m.gen([rid, did, rid], None, N_NEW)
        g1 = m.gen2([rid, did, rid], None, N_NEW)
        worst["gen_tok"] += int((g0["tokens"] != g1["tokens"]).sum())
        worst["gen_top1"] += int((g0["top1"] != g1["top1"]).sum())
        g2 = m.gen2([rid, did, rid], None, N_NEW, force=(torch.tensor([False, True, False]), c))
        worst["force_top1"] += int((g2["top1"][:, :K] != g1["top1"][:, :K]).sum())
        # 2. self-patch from the clean row at every prompt and answer position is the identity
        rows = [dict(cond="iiip", l=l) for l in LAYERS]
        B = 1 + len(rows)
        g = m.gen2([rid] * B, build_patch(rows, B, W, S, 1, lambda rw: 0), N_NEW)
        worst["self_tok"] += int((g["tokens"][1:] != g["tokens"][0:1]).sum())
        worst["self_top1"] += int((g["top1"][1:] != g["top1"][0:1]).sum())
        # 3. exact write and causality; 4. the all-position patch reproduces the donor row above l
        rows = [dict(cond=cc, l=l) for cc in CONDS for l in LAYERS]
        B = 2 + len(rows)
        g = m.gen2([rid, did] + [rid] * len(rows), build_patch(rows, B, W, S, 2, lambda rw: 1), N_NEW,
                   force=(torch.tensor([False, True] + [False] * len(rows)), c), keep_layers=list(range(L)))
        kp = g["kept"]
        oc0 = outcome(m, rit, g["tokens"][0], g["top1"][0], keep_top1=True)
        ocd = outcome(m, dit, g["tokens"][1], g["top1"][1], keep_top1=True)
        for j, rw in enumerate(rows):
            b, l, ps = 2 + j, rw["l"], S[rw["cond"]]
            worst["write"] = max(worst["write"], float((kp[l][b, ps].float() - kp[l][1, ps].float()).abs().max()))
            first = min(ps)
            if first > 0:
                worst["causal"] = max(worst["causal"], max(float((kp[k][b, :first].float() - kp[k][0, :first].float()).abs().max())
                                                           for k in range(L)))
            ks = [k for k in range(g["top1"].shape[1]) if n - 1 + k < first]  # steps read before the patch
            worst["causal_top1"] += int((g["top1"][b, ks] != g["top1"][0, ks]).sum())
            if rw["cond"] == "iiip":
                worst["splice_top1"] = max(worst["splice_top1"],
                                           float((g["top1"][b, :K, l:] != g["top1"][1, :K, l:]).float().mean()))
                ocb = outcome(m, rit, g["tokens"][b], g["top1"][b])
                worst["splice_D_rows"] += 1
                worst["splice_D_match"] += int(ocb["settle"] == splice(oc0["top1"], ocd["top1"], l))
        log(f"{grp[0]} <- {grp[1]}: {worst}")
    log("selftest", worst)
    assert worst["gen_tok"] == 0 and worst["gen_top1"] == 0, "engine differs from PatchModel.gen"
    assert worst["force_top1"] == 0, "forcing the row's own answer changed it"
    assert worst["self_tok"] == 0 and worst["self_top1"] == 0, "self-patch is not the identity"
    assert worst["write"] == 0.0, "the patch did not write exactly the source row's state"
    assert worst["causal"] == 0.0 and worst["causal_top1"] == 0, "a patch changed earlier positions"
    assert worst["splice_top1"] == 0.0, "all-position patch does not reproduce the donor's lens above l"
    assert worst["splice_D_match"] == worst["splice_D_rows"], "all-position patch D differs from the splice"
    log("SELFTEST PASSED")
    sys.exit(0)

# ===================================================================== pair2
if STAGE == "pair2":
    pool = {r["id"]: r for r in read_jsonl(POOL)}
    S0 = json.load(open(os.path.join(OLD, "res", "pairs.json"), encoding="utf-8"))["S"]
    out, dropped = [], 0
    for x in S0:
        c = pool[x["r"]]["cont"]
        far = [d for d in x["far"] if pool[d]["cont"] == c]
        null = [d for d in x["null"] if pool[d]["cont"] == c]
        dropped += len(x["far"]) + len(x["null"]) - len(far) - len(null)
        if far and null:
            out.append(dict(x, far=far, null=null, K=len(c)))
    json.dump(dict(S=out), open(os.path.join(RES, "pairs2.json"), "w"), indent=1)
    log(f"arm S2: {len(out)} of {len(S0)} recipients (early {sum(x['s'] == 1 for x in out)}, late "
        f"{sum(x['s'] == -1 for x in out)}); donors dropped for unequal answer tokens {dropped}; far "
        f"{sum(len(x['far']) for x in out)}, null {sum(len(x['null']) for x in out)}; distinct items "
        f"{len({x['r'] for x in out} | {d for x in out for d in x['far'] + x['null']})}; "
        f"K counts {dict(zip(*np.unique([x['K'] for x in out], return_counts=True)))}")
    log("DONE")
    sys.exit(0)

# ================================================================== patchS2
assert STAGE == "patchS2", STAGE
pool = {r["id"]: r for r in read_jsonl(POOL)}
pairs = json.load(open(os.path.join(RES, "pairs2.json"), encoding="utf-8"))["S"]
path = os.path.join(RES, "patchS2" + ("_smoke" if N_LIMIT else "") + ".jsonl")
done = {r["r"] for r in read_jsonl(path)}
recs = pairs[:N_LIMIT] if N_LIMIT else pairs
todo = [x for x in recs if x["r"] not in done]
log(f"arm S2: {len(todo)} of {len(recs)} recipients to run; max rows {MAX_ROWS}")
with open(path, "a", encoding="utf-8") as f:
    for nn, pr in enumerate(todo):
        t_it = time.time()
        rit = g_item(pr["r"])
        rid, start, src = layout(m, rit)
        n = len(rid)
        c = pool[pr["r"]]["cont"]
        K = len(c)
        W = n + K - 1
        S = pos_sets(n, start, src, K)
        donors = [(d, "far") for d in pr["far"]] + [(d, "null") for d in pr["null"]]
        dids = []
        for d, _ in donors:
            ids_d, st_d, src_d = layout(m, g_item(d))
            assert len(ids_d) == n and st_d == start and src_d == src, "donor layout differs"
            assert pool[d]["cont"] == c, "donor answer tokens differ"
            dids.append(ids_d)
        nd = len(dids)
        rows = [dict(donor=d, role=role, cond=cc, l=l, di=k) for k, (d, role) in enumerate(donors)
                for cc in CONDS for l in LAYERS]
        out, clean, donclean = [], None, {}
        step = MAX_ROWS - 1 - nd
        for c0 in range(0, len(rows), step):
            ch = rows[c0:c0 + step]
            B = 1 + nd + len(ch)
            force = (torch.tensor([False] + [True] * nd + [False] * len(ch)), c)
            g = m.gen2([rid] + dids + [rid] * len(ch), build_patch(ch, B, W, S, 1 + nd, lambda rw: 1 + rw["di"]),
                       N_NEW, force=force)
            oc0 = outcome(m, rit, g["tokens"][0], g["top1"][0], keep_top1=True)
            if clean is None:
                clean = oc0
            elif oc0["cont"] != clean["cont"] or oc0["settle"] != clean["settle"]:
                clean["chunk_mismatch"] = clean.get("chunk_mismatch", 0) + 1
            for k, (d, _) in enumerate(donors):
                od = outcome(m, g_item(d), g["tokens"][1 + k], g["top1"][1 + k], keep_top1=True)
                od["cont_same"] = int(od["cont"] == c)  # the donor's own argmax under forcing is the shared answer
                od.pop("cont")
                if d not in donclean:
                    donclean[d] = od
                elif od["settle"] != donclean[d]["settle"]:
                    donclean[d]["chunk_mismatch"] = donclean[d].get("chunk_mismatch", 0) + 1
            for j, rw in enumerate(ch):
                b = 1 + nd + j
                oc = outcome(m, rit, g["tokens"][b], g["top1"][b])
                out.append(dict(rw, ans=oc["ans"], ok=oc["correct"], D=oc["D"], D_gen=oc["D_gen"], n_dig=oc["n_dig"],
                                settle=oc["settle"], same_tok=int(oc["cont"] == c)))
        clean["cont_same"] = int(clean.pop("cont") == c)
        f.write(json.dumps(dict(r=pr["r"], s=pr["s"], K=K, clean=clean, donclean=donclean,
                                D_pool=pool[pr["r"]]["D"],
                                donD={d: pool[d]["D"] for d, _ in donors},
                                donDgen={d: pool[d]["D_gen"] for d, _ in donors},
                                donSettle={d: pool[d]["settle"] for d, _ in donors}, rows=out)) + "\n")
        f.flush()
        if nn % 10 == 0:
            log(f"S2 {nn}/{len(todo)} {pr['r']} {time.time() - t_it:.1f}s ({len(rows)} rows)")
log("DONE")
