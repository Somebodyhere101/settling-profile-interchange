"""Paper 8 patching arm (patch/PREDICTIONS.md): where does settling depth live?

python run_patch.py <stage> [n_recipients]

selftest  engine checks (no target contrast is computed): the unpatched engine reproduces p8lib's gen8;
          a self-patch is the identity; patches leave earlier positions untouched and write exactly the
          donor's state; an all-position patch at layer 7 reproduces the donor's answer.
pool      clean outcomes (answer, settling depth D, answer tokens) for paper 4's M items (items.json d3)
          and for the same-format generated pool used by arm S (every a * b with 11 <= a, b <= 99 and a
          four-digit product). Rows of a batch are different prompts of equal length.
pair      the preregistered pairing (§3): arm P (M items, early vs late, by answer digit count) and
          arm S (generated pool, same product). Writes res/pairs.json; prints counts only.
patchP    arm P: for each recipient, its far and null donors' block outputs patched at (i) the question
patchS    tokens, (ii) the reading position onward, (ii-r) the reading position only and (iii) all
          prompt positions, at layers 7/14/21/24. Outcomes: answer, D, D over generated positions, and
          (arm P) the teacher-forced log-odds of the donor's answer against the recipient's.
With n_recipients, patch stages write res/*_smoke_* (timing only).
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "..", "code")
sys.path.insert(0, CODE)

from patchlib import PatchModel  # noqa: E402
from specs import D3, ITEMS  # noqa: E402

STAGE = sys.argv[1]
N_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
KEY = "sm2-1.7b"
LAYERS = [6, 12, 18, 21]
CONDS = ["i", "ii", "iir", "iii"]
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
N_NEW = D3.n_new_mode["direct"]
SEED = 8
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = D3()
P = spec.P  # paper 4's parsers (cr4_parsers.py)


def m_items():
    I = ITEMS["d3"]
    return [x for x in {x["id"]: x for x in I["A"] + I["oracle"]}.values() if x["set"] == "M"]


def g_items():
    out = []
    for a in range(11, 100):
        for b in range(11, 100):
            if a * b >= 1000:
                out.append(dict(id=f"G:{a}*{b}", set="G", a=a, b=b, op="*", answer=a * b,
                                prompt=f"What is {a} * {b}?", prefill=f"{a} * {b} = "))
    return out


def layout(m, it):
    """Prompt ids, reading position (paper 6 / pilot: last token of the CR detector's chat template) and
    the question-token positions (the user-message content, as run8.py's source_positions)."""
    text = m.chat(it["prompt"]) + it["prefill"]
    ids, start = spec.prompt_ids(m, it, "direct")
    enc = m.tok(text, add_special_tokens=not text.startswith(m.tok.bos_token or "\x00"), return_offsets_mapping=True)
    assert enc["input_ids"] == ids
    a = text.index(it["prompt"])
    b = a + len(it["prompt"])
    src = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < b and e > a]
    assert src and max(src) < start
    return ids, start, src


def pos_sets(n, start, src):
    return dict(i=list(src), ii=list(range(start, n)), iir=[start], iii=list(range(n)))


def outcome(m, it, toks, top1):
    """D3's direct-mode outcome plus D over the digits read at generated positions (digits 2..n) and the
    answer tokens (through the last digit)."""
    oc = spec.outcome(m, it, "direct", toks, None, top1)
    tt = m.trunc(toks)
    ans, idx = P.answer_span(m.tok, tt)
    st = [P.settle_top1(top1[i]) for i in idx]
    oc["D_gen"] = float(np.mean(st[1:])) / m.L if len(st) > 1 else None
    oc["settle"] = st
    oc["cont"] = [int(t) for t in tt[:idx[-1] + 1]] if idx else []
    oc.pop("text", None)
    return oc


# ================================================================= selftest
if STAGE == "selftest":
    m = PatchModel(KEY)
    L = m.L
    its = m_items()
    rng = np.random.default_rng(0)
    pick = [its[j] for j in rng.choice(len(its), 6, replace=False)]
    lay = [layout(m, x) for x in pick]
    n, start, src = len(lay[0][0]), lay[0][1], lay[0][2]
    assert all(len(x[0]) == n and x[1] == start and x[2] == src for x in lay), "layouts differ"
    S = pos_sets(n, start, src)
    log(f"n_prompt={n} reading position={start} question tokens={src}")
    worst = dict(gen8_tok=0, gen8_top1=0, self_tok=0, self_top1=0.0, causal=0.0, write=0.0, other=0.0, iii_donor=0)
    for r in range(3):
        rid, did = lay[r][0], lay[r + 3][0]
        # 1. the unpatched engine reproduces p8lib's gen8 (same prompt in every row)
        B = 4
        g0 = m.gen8(rid, {}, torch.zeros(B, n), torch.zeros(B), N_NEW, lens=True)
        g1 = m.gen([rid] * B, None, N_NEW)
        worst["gen8_tok"] += int((g0["tokens"] != g1["tokens"]).sum())
        worst["gen8_top1"] += int((g0["top1"] != g1["top1"]).sum())
        # 2. a self-patch (the recipient's own states, from a batch of other prompts) is the identity
        Hs = m.states([rid, did, lay[(r + 1) % 6][0]], LAYERS)
        rows = [(c, l) for c in CONDS for l in LAYERS]
        patch = {}
        for l in LAYERS:
            mask = torch.zeros(len(rows) + 1, n, dtype=torch.bool)
            for b, (c, ll) in enumerate(rows, 1):
                if ll == l:
                    mask[b, S[c]] = True
            patch[l] = (mask, Hs[l][0:1].expand(len(rows) + 1, -1, -1))
        g = m.gen([rid] * (len(rows) + 1), patch, N_NEW)
        worst["self_tok"] += int((g["tokens"][1:] != g["tokens"][0:1]).sum())
        worst["self_top1"] = max(worst["self_top1"], float((g["top1"][1:] != g["top1"][0:1]).float().mean()))
        # 3. causality and exact writes: donor patch at (c, l); capture every layer at every prompt position
        for c in ("i", "ii", "iir"):
            for l in LAYERS:
                mask = torch.zeros(2, n, dtype=torch.bool)
                mask[1, S[c]] = True
                pt = {l: (mask, torch.stack([Hs[l][0], Hs[l][1]]))}
                _, cap = m.cont_lp([rid, rid], pt, [[0], [0]], cap_layers=list(range(L)))
                first = min(S[c])
                worst["causal"] = max(worst["causal"], max(float((cap[k][1, :first] - cap[k][0, :first]).abs().max())
                                                           for k in range(L)))
                worst["write"] = max(worst["write"], float((cap[l][1, S[c]] - Hs[l][1, S[c]].float()).abs().max()))
                other = [t for t in range(n) if t not in S[c]]
                worst["other"] = max(worst["other"], float((cap[l][1, other] - cap[l][0, other]).abs().max()))
        # 4. the all-position patch at layer 7 produces the donor's own answer tokens
        gd = m.gen([did], None, N_NEW)
        mask = torch.zeros(2, n, dtype=torch.bool)
        mask[1] = True
        gp = m.gen([rid, rid], {7: (mask, torch.stack([Hs[7][0], Hs[7][1]]))}, N_NEW)
        cd = outcome(m, pick[r + 3], gd["tokens"][0], gd["top1"][0])["cont"]
        cp = outcome(m, pick[r + 3], gp["tokens"][1], gp["top1"][1])["cont"]
        worst["iii_donor"] += int(cd != cp)
        # 5. teacher-forced log-prob of the clean continuation matches the generation's first-step log-prob
        cr = outcome(m, pick[r], g1["tokens"][0], g1["top1"][0])["cont"]
        lp1 = m.cont_lp([rid], None, [cr[:1]])[0]
        worst["lp_first"] = max(worst.get("lp_first", 0.0), abs(lp1 - float(g1["lp0"][0, cr[0]])))
        log(f"{pick[r]['id']} <- {pick[r + 3]['id']}: {worst}")
    log("selftest", worst)
    assert worst["gen8_tok"] == 0 and worst["gen8_top1"] == 0, "engine differs from p8lib gen8"
    assert worst["self_tok"] == 0 and worst["self_top1"] < 0.01, "self-patch is not the identity"
    assert worst["causal"] == 0.0, "a patch changed earlier positions"
    assert worst["write"] == 0.0 and worst["other"] == 0.0, "the patch did not write exactly the donor's state"
    assert worst["iii_donor"] == 0, "all-position patch at layer 7 does not reproduce the donor's answer"
    assert worst["lp_first"] < 0.05, "teacher-forced log-prob does not match generation"
    log("SELFTEST PASSED")
    sys.exit(0)

m = PatchModel(KEY)
L = m.L


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


# ===================================================================== pool
if STAGE == "pool":
    path = os.path.join(RES, "pool_clean.jsonl")
    done = {r["id"] for r in read_jsonl(path)}
    items = m_items() + g_items()
    todo = [x for x in items if x["id"] not in done]
    log(f"pool: {len(todo)} of {len(items)} items")
    lay0 = layout(m, items[0])
    with open(path, "a", encoding="utf-8") as f:
        for c0 in range(0, len(todo), MAX_ROWS):
            ch = todo[c0:c0 + MAX_ROWS]
            lays = [layout(m, x) for x in ch]
            assert all(len(x[0]) == len(lay0[0]) and x[1] == lay0[1] and x[2] == lay0[2] for x in lays)
            g = m.gen([x[0] for x in lays], None, N_NEW)
            for b, it in enumerate(ch):
                oc = outcome(m, it, g["tokens"][b], g["top1"][b])
                f.write(json.dumps(dict(id=it["id"], set=it["set"], a=it["a"], b=it["b"], answer=it["answer"], **oc)) + "\n")
            f.flush()
            if (c0 // MAX_ROWS) % 10 == 0:
                log(f"pool {c0 + len(ch)}/{len(todo)}")
    # agreement with the pilot's batch-1 clean D for the M items (reported, not a gate)
    pil = {r["id"]: r for r in read_jsonl(os.path.join(CODE, "res", "settle_q3-1.7b_clean.jsonl"))}
    mine = {r["id"]: r for r in read_jsonl(path)}
    same = [mine[i]["D"] == pil[i]["direct"]["D"] for i in pil if i in mine and pil[i]["set"] == "M"]
    log(f"M items: D identical to the pilot's clean stage in {sum(same)}/{len(same)}")
    log("DONE")
    sys.exit(0)

# ===================================================================== pair
if STAGE == "pair":
    pool = {r["id"]: r for r in read_jsonl(os.path.join(RES, "pool_clean.jsonl"))}
    rng = np.random.default_rng(SEED)
    q = lambda i: pool[i]["D"] * L  # depth in layers
    ok = lambda r: r["correct"] == 1 and r["D"] is not None
    out = dict(P=[], S=[])
    # ---- arm P: paper 4's M items, by answer digit count
    M = [i for i, r in pool.items() if r["set"] == "M" and ok(r) and r["n_dig"] in (3, 4)]
    for nd in (3, 4):
        S_ = sorted(i for i in M if pool[i]["n_dig"] == nd)
        med = float(np.median([q(i) for i in S_]))
        early = [i for i in S_ if q(i) <= med - 0.5 + 1e-9]
        late = [i for i in S_ if q(i) >= med + 0.5 - 1e-9]
        for tail, opp, s in ((early, late, 1), (late, early, -1)):
            for r in tail:
                far = [d for d in opp if pool[d]["answer"] != pool[r]["answer"]]
                far = [far[j] for j in rng.permutation(len(far))[:3]]
                cand = [d for d in S_ if d != r and d not in far and pool[d]["answer"] != pool[r]["answer"]]
                exact = [d for d in cand if abs(q(d) - q(r)) <= 0.125]  # equal depth (D*L on a 1/4 or 1/3 grid)
                near = [d for d in cand if 0.125 < abs(q(d) - q(r)) <= 0.34]  # one grid step, only to fill up
                null = [exact[j] for j in rng.permutation(len(exact))[:3]]
                null += [near[j] for j in rng.permutation(len(near))[:3 - len(null)]]
                if far and null:
                    out["P"].append(dict(r=r, s=s, nd=nd, far=far, null=null, med=med))
    # ---- arm S: generated pool, same product
    G = [i for i, r in pool.items() if r["set"] == "G" and ok(r)]
    byp = {}
    for i in G:
        byp.setdefault(pool[i]["answer"], []).append(i)
    elig = []
    for prod, grp in sorted(byp.items()):
        for r in sorted(grp):
            up = [d for d in grp if q(d) - q(r) >= 1.0 - 1e-9]
            dn = [d for d in grp if q(r) - q(d) >= 1.0 - 1e-9]
            null = [d for d in grp if d != r and abs(q(d) - q(r)) <= 0.125]
            if not (up or dn) or not null:
                continue
            if len(up) != len(dn):
                side, s = (up, 1) if len(up) > len(dn) else (dn, -1)
            else:
                gu = max([q(d) - q(r) for d in up], default=0)
                gd = max([q(r) - q(d) for d in dn], default=0)
                side, s = (up, 1) if gu >= gd else (dn, -1)
            elig.append(dict(r=r, s=s, prod=prod, side=sorted(side), null=sorted(null)))
    order = rng.permutation(len(elig))
    per_prod = {}
    for j in order:
        e = elig[j]
        if per_prod.get(e["prod"], 0) >= 2 or len(out["S"]) >= 150:
            continue
        per_prod[e["prod"]] = per_prod.get(e["prod"], 0) + 1
        far = [e["side"][k] for k in rng.permutation(len(e["side"]))[:2]]
        null = [e["null"][k] for k in rng.permutation(len(e["null"]))[:2]]
        out["S"].append(dict(r=e["r"], s=e["s"], nd=4, far=far, null=null))
    out["S_eligible"] = len(elig)
    out["S_run"] = len(out["S"]) >= 40
    json.dump(out, open(os.path.join(RES, "pairs.json"), "w"), indent=1)
    for arm in ("P", "S"):
        X = out[arm]
        log(f"arm {arm}: {len(X)} recipients (early {sum(x['s'] == 1 for x in X)}, late {sum(x['s'] == -1 for x in X)}); "
            f"far donors {sum(len(x['far']) for x in X)}, null donors {sum(len(x['null']) for x in X)}; "
            f"distinct items {len({x['r'] for x in X} | {d for x in X for d in x['far'] + x['null']})}")
    log(f"arm S: {out['S_eligible']} eligible; runs: {out['S_run']}")
    log("DONE")
    sys.exit(0)

# ==================================================================== patch
assert STAGE in ("patchP", "patchS"), STAGE
ARM = STAGE[-1]
pool = {r["id"]: r for r in read_jsonl(os.path.join(RES, "pool_clean.jsonl"))}
pairs = json.load(open(os.path.join(RES, "pairs.json"), encoding="utf-8"))
if ARM == "S" and not pairs["S_run"]:
    log("arm S: fewer than 40 eligible recipients; not run (PREDICTIONS.md §3)")
    sys.exit(0)
byid = {x["id"]: x for x in m_items() + g_items()}
PRE = os.path.join(RES, f"patch{ARM}" + ("_smoke" if N_LIMIT else ""))
path = PRE + ".jsonl"
done = {r["r"] for r in read_jsonl(path)}
recs = pairs[ARM][:N_LIMIT] if N_LIMIT else pairs[ARM]
todo = [x for x in recs if x["r"] not in done]
log(f"arm {ARM}: {len(todo)} of {len(recs)} recipients to run; max rows {MAX_ROWS}")
with open(path, "a", encoding="utf-8") as f:
    for nn, pr in enumerate(todo):
        t_it = time.time()
        rit = byid[pr["r"]]
        rid, start, src = layout(m, rit)
        n = len(rid)
        S = pos_sets(n, start, src)
        donors = [(d, "far") for d in pr["far"]] + [(d, "null") for d in pr["null"]]
        dids = []
        for d, _ in donors:
            ids_d, st_d, src_d = layout(m, byid[d])
            assert len(ids_d) == n and st_d == start and src_d == src, "donor layout differs"
            dids.append(ids_d)
        H = m.states(dids, LAYERS)  # donor block outputs [nd, n, d] per layer
        rows = [dict(donor=d, role=role, cond=c, l=l, di=k) for k, (d, role) in enumerate(donors)
                for c in CONDS for l in LAYERS]
        out = []
        clean = None
        step = MAX_ROWS - 1
        for c0 in range(0, len(rows), step):
            ch = rows[c0:c0 + step]
            B = len(ch) + 1
            patch = {}
            for l in LAYERS:
                mask = torch.zeros(B, n, dtype=torch.bool)
                don = torch.zeros(B, n, m.d, dtype=torch.bfloat16, device=m.dev)
                for b, rw in enumerate(ch, 1):
                    if rw["l"] == l:
                        mask[b, S[rw["cond"]]] = True
                        don[b] = H[l][rw["di"]]
                if mask.any():
                    patch[l] = (mask, don)
            g = m.gen([rid] * B, patch, N_NEW)
            oc0 = outcome(m, rit, g["tokens"][0], g["top1"][0])
            if clean is None:
                clean = oc0
            elif oc0["cont"] != clean["cont"] or oc0["D"] != clean["D"]:
                clean["chunk_mismatch"] = clean.get("chunk_mismatch", 0) + 1
            ocs = [outcome(m, rit, g["tokens"][b], g["top1"][b]) for b in range(1, B)]
            if ARM == "P":  # log-odds of the donor's answer against the recipient's, teacher-forced
                ans_r = pool[pr["r"]]["cont"]
                cd = [pool[rw["donor"]]["cont"] for rw in ch]
                lpd = m.cont_lp([rid] * B, patch, [ans_r] + cd)
                lpr = m.cont_lp([rid] * B, patch, [ans_r] * B)
            for b, (rw, oc) in enumerate(zip(ch, ocs), 1):
                rec = dict(rw, ans=oc["ans"], ok=oc["correct"], D=oc["D"], D_gen=oc["D_gen"], n_dig=oc["n_dig"],
                           don_ans=int(oc["ans"] == pool[rw["donor"]]["answer"]))
                if ARM == "P":
                    rec["lm"] = lpd[b] - lpr[b]  # log p(donor answer) - log p(recipient answer), patched
                out.append(rec)
        if ARM == "P":  # the same log-odds in the clean recipient, per donor
            cd = [pool[d]["cont"] for d, _ in donors]
            lc_d = m.cont_lp([rid] * len(cd), None, cd)
            lc_r = m.cont_lp([rid], None, [pool[pr["r"]]["cont"]])[0]
            lm0 = {d: v - lc_r for (d, _), v in zip(donors, lc_d)}
            for rec in out:
                rec["dlm"] = rec["lm"] - lm0[rec["donor"]]
        clean.pop("cont", None)
        f.write(json.dumps(dict(r=pr["r"], s=pr["s"], nd=pr["nd"], clean=clean,
                                D_pool=pool[pr["r"]]["D"],
                                donD={d: pool[d]["D"] for d, _ in donors}, rows=out)) + "\n")
        f.flush()
        if nn % 10 == 0:
            log(f"{ARM} {nn}/{len(todo)} {pr['r']} {time.time() - t_it:.1f}s ({len(rows)} rows)")
log("DONE")
