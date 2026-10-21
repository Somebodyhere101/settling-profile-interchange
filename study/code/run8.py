"""Paper 8 pilot: where can the course of computation be pushed? (PREDICTIONS.md)

python run8.py <settle|flip> <model key> <selftest|clean|push> [n_items]

settle   settling depth of the answer digits (paper 6's D3/D4 target) on paper 4's format-matched
         arithmetic items (paper 6 items.json "d3"), direct mode with the "a * b = " prefill.
flip     flip to the user's stated wrong option on ARC (paper 6's D8 target; positive control).

selftest  the masked engine with paper 6's mask reproduces p6lib.Model.generate / tf_kl; source-only
          pushes leave every earlier position unchanged. Stops on any failure.
clean     reading-position and source-position hidden states at the three layers, clean outcomes
          (Check A and oracle items, single rows).
push      fresh oracles (reading position: o; source tokens: os) on the oracle split; then, on the
          Check D items, o / os / 12 Gaussian directions pushed under three position conditions at
          three layers (rows of one batch share the prompt; row 0 clean). Outcomes, KL over the
          output window, and the persistence of each write at the answer positions.

Conditions: a = reading position onward incl. generated tokens (paper 6); b = every position from 1
on (all prompt tokens but the first, plus generated tokens); c = the user-message tokens only.
Resumable per item. With n_items, writes to <key>_smoke_* instead.
"""

import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch

from p6lib import Model, gaussian, oracle, unit
from p8lib import P8Model
from specs import D3, ITEMS
from spec_d8 import D8, PREFILL, user_text

TARGET, KEY, STAGE = sys.argv[1], sys.argv[2], sys.argv[3]
N_LIMIT = int(sys.argv[4]) if len(sys.argv) > 4 else None
HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config8.json"), encoding="utf-8"))
if os.environ.get("MAX_ROWS"):
    CFG["max_rows"] = int(os.environ["MAX_ROWS"])
os.environ["P6_PHASE"] = "main"
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
PRE = os.path.join(RES, f"{TARGET}_{KEY}" + ("_smoke" if N_LIMIT else ""))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ target
if TARGET == "settle":
    spec = D3()
    I = ITEMS["d3"]
    SETS = dict(A=I["A"], oracle=I["oracle"], D=I["D"])
    MODE, LENS, N_NEW, BINARY = "direct", True, D3.n_new_mode["direct"], False

    def prompt_text(m, it):
        return m.chat(it["prompt"]) + it["prefill"], it["prompt"]

    def T_of(out, clean_rec):
        return out.get("D")
else:
    spec = D8()
    I = json.load(open(os.path.join(HERE, "data", "items_d8.json"), encoding="utf-8"))
    SETS = dict(A=I["A"], oracle=I["oracle"], D=I["D"])
    MODE, LENS, N_NEW, BINARY = "belief", False, D8.n_new_mode["belief"], True

    def prompt_text(m, it):
        u = user_text(it, True)
        return m.chat(u) + PREFILL, u

    def T_of(out, clean_rec):  # flips only count on items answered correctly without the belief
        return out.get("flip") if clean_rec["plain"]["correct"] else None

MC = CFG["models"][KEY][TARGET]
ELL, G_PUB = MC["ell"], MC["G"]


def layers_for(L):
    return [ELL - round(L / 4), ELL, ELL + math.ceil((L - 1 - ELL) / 2)]


def source_positions(m, it):
    """Token indices (in the prompt ids) covering the user-message content."""
    text, content = prompt_text(m, it)
    ids, start = spec.prompt_ids(m, it, MODE)
    enc = m.tok(text, add_special_tokens=not text.startswith(m.tok.bos_token or "\x00"), return_offsets_mapping=True)
    assert enc["input_ids"] == ids, "prompt text does not reproduce the spec's ids"
    a = text.index(content)
    b = a + len(content)
    src = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if e > s and s < b and e > a]
    assert src and max(src) < start, "source tokens must lie before the reading position"
    return ids, start, src


def strip(oc):
    return {k: v for k, v in oc.items() if k != "text"}


def masks(cond, n, start, src):
    pm = torch.zeros(n)
    if cond == "a":
        pm[start:] = 1
        g = 1.0
    elif cond == "b":
        pm[1:] = 1
        g = 1.0
    elif cond == "c":
        pm[src] = 1
        g = 0.0
    else:  # clean
        g = 0.0
    return pm, g


# ================================================================= selftest
if STAGE == "selftest":
    # paper 6's engine first (its hook only knows `start`), then the masked engine on the same rows
    base = Model(KEY)
    L = base.L
    LAY = layers_for(L)
    log(f"{base.name} L={L} layers={LAY}")
    rng = np.random.default_rng(0)
    byid = {x["id"]: x for x in SETS["A"] + SETS["oracle"]}
    B = 6
    cases = []
    for iid in SETS["D"][:3]:
        it = byid[iid]
        ids, start, src = source_positions(base, it)
        log(f"{iid}: n_prompt={len(ids)} start={start} source={src} -> {base.tok.decode([ids[i] for i in src])!r}")
        add = {l: torch.zeros(B, base.d) for l in LAY}
        for b in range(1, B):  # one layer per row, as in paper 6's batches
            add[LAY[b % 3]][b] = torch.tensor(unit(rng.standard_normal(base.d)) * G_PUB * b, dtype=torch.float32)
        r0 = base.generate(ids, start, add, N_NEW, lens=LENS)
        cont = base.trunc(r0["tokens"][0])
        k0 = base.tf_kl(ids, start, add, cont)["kl_all"]
        cases.append((it, ids, start, src, add, r0, cont, k0))
    del base  # p6lib's hooks hold the model in a reference cycle: collect it before loading the second copy
    gc.collect()
    torch.cuda.empty_cache()
    m = P8Model(KEY)
    worst = dict(lp0=0.0, kl=0.0, tok=0, causal=0.0)
    for it, ids, start, src, add, r0, cont, k0 in cases:
        pm = torch.zeros(B, len(ids))
        pm[:, start:] = 1
        gm = torch.ones(B)
        r1 = m.gen8(ids, add, pm, gm, N_NEW, lens=LENS)
        n_tok = r0["tokens"].shape[1]
        worst["tok"] += int((r0["tokens"] != r1["tokens"]).sum()) if r1["tokens"].shape[1] == n_tok else 999
        worst["lp0"] = max(worst["lp0"], float((r0["lp0"] - r1["lp0"]).abs().max()))
        k1, _ = m.tf8(ids, start, add, pm, gm, cont, [LAY[0]], [start])
        worst["kl"] = max(worst["kl"], max(abs(a - b) / (1e-3 + abs(a)) for a, b in zip(k0, k1)))
        if LENS:
            assert torch.equal(r0["top1"], r1["top1"]), "lens top-1 differs"
        # causality: a source-only push leaves positions before the first source token unchanged ...
        pmc = torch.zeros(B, len(ids))
        pmc[:, src] = 1
        early = list(range(min(src)))
        _, cap = m.tf8(ids, start, add, pmc, torch.zeros(B), cont, list(range(L)), early)
        _, cap0 = m.tf8(ids, start, {}, pmc, torch.zeros(B), cont, list(range(L)), early)
        worst["causal"] = max(worst["causal"], max(float((cap[l] - cap0[l]).abs().max()) for l in range(L)))
        # ... and does reach the reading position after the push layer
        _, capr = m.tf8(ids, start, add, pmc, torch.zeros(B), cont, [L - 1], [start])
        _, capr0 = m.tf8(ids, start, {}, pmc, torch.zeros(B), cont, [L - 1], [start])
        log(f"  {it['id']}: source push reaches the reading position, max |dh| at the last layer = "
            f"{float((capr[L - 1] - capr0[L - 1]).abs().max()):.2f}")
        # the write itself at the push layer and the reading position: the full vector under (a), nothing under (c)
        for l in LAY:
            _, ca = m.tf8(ids, start, add, pm, gm, cont, [l], [start])
            _, cc = m.tf8(ids, start, add, pmc, torch.zeros(B), cont, [l], [start])
            _, c0 = m.tf8(ids, start, {}, pm, gm, cont, [l], [start])
            want = add[l].to(m.dev)
            on = want.norm(dim=-1) > 0  # rows pushed at this layer (bf16 rounding of h + v: compare directions and sizes)
            got = (ca[l][:, 0] - c0[l][:, 0])[on]
            cos = torch.nn.functional.cosine_similarity(got, want[on], dim=-1)
            ratio = got.norm(dim=-1) / want[on].norm(dim=-1)
            err_a = float(torch.maximum(1 - cos, (ratio - 1).abs()).max())
            worst["write_a"] = max(worst.get("write_a", 0.0), err_a)
            # only rows pushed at this layer: a source write made at an earlier layer has legitimately reached it
            worst["write_c"] = max(worst.get("write_c", 0.0), float((cc[l][:, 0] - c0[l][:, 0])[on].abs().max()))
    log("selftest", worst)
    assert worst["causal"] < 1.0, "a source-only push changed earlier positions"
    assert worst["write_a"] < 0.05 and worst["write_c"] < 1.0, "the write at the push layer is not what the mask says"
    assert worst["tok"] == 0 and worst["lp0"] < 1e-3 and worst["kl"] < 1e-3, "masked engine differs from p6lib"
    log("SELFTEST PASSED")
    sys.exit(0)

m = P8Model(KEY)
L = m.L
LAY = layers_for(L)
log(f"{m.name} L={L} d={m.d} target={TARGET} stage={STAGE} published layer {ELL}, layers {LAY}, G={G_PUB:.2f}")


def done_ids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")}


# ==================================================================== clean
if STAGE == "clean":
    path = PRE + "_clean.jsonl"
    done = done_ids(path)
    items = list({x["id"]: x for x in SETS["A"] + SETS["oracle"]}.values())
    if N_LIMIT:
        dset = set(SETS["D"][:N_LIMIT])
        items = [x for x in items if x["id"] in dset] + [x for x in SETS["oracle"][:max(60, N_LIMIT)]]
        items = list({x["id"]: x for x in items}.values())
    todo = [x for x in items if x["id"] not in done]
    log(f"clean: {len(todo)} of {len(items)} items")
    hpath = PRE + "_clean_h.npz"
    Hs = dict(np.load(hpath)) if os.path.exists(hpath) else {}
    pending = []

    def save(f):  # hidden states first, then the records, so that every record on disk has its states
        np.savez(hpath, **Hs)
        for line in pending:
            f.write(line)
        f.flush()
        pending.clear()
    with open(path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            ids, start, src = source_positions(m, it)
            a = m.acts(ids, LAY, [start] + src)
            for l in LAY:
                Hs[f"{it['id']}|{l}|r"] = a[l][0].astype(np.float32)
                Hs[f"{it['id']}|{l}|s"] = a[l][1:].astype(np.float64).mean(0).astype(np.float32)
            rec = dict(id=it["id"], set=it.get("set"), n_prompt=len(ids), start=start, src=src)
            r = m.gen8(ids, {}, torch.zeros(1, len(ids)), torch.zeros(1), N_NEW, lens=LENS)
            rec[MODE] = strip(spec.outcome(m, it, MODE, r["tokens"][0], r["lp0"][0], r["top1"][0] if LENS else None))
            if TARGET == "flip":
                ids_p, _ = spec.prompt_ids(m, it, "plain")
                rp = m.gen8(ids_p, {}, torch.zeros(1, len(ids_p)), torch.zeros(1), D8.n_new_mode["plain"])
                rec["plain"] = strip(spec.outcome(m, it, "plain", rp["tokens"][0], rp["lp0"][0]))
            rec["T"] = T_of(rec[MODE], rec)
            pending.append(json.dumps(rec) + "\n")
            if n % 50 == 0:
                log(f"clean {n}/{len(todo)} {it['id']} n_src={len(src)} T={rec['T']}")
                save(f)
        save(f)
    log("DONE")
    sys.exit(0)

# ===================================================================== push
assert STAGE == "push", STAGE
clean = {}
for l_ in open(PRE + "_clean.jsonl", encoding="utf-8"):
    r_ = json.loads(l_)
    clean[r_["id"]] = r_
Hs = dict(np.load(PRE + "_clean_h.npz"))
orc = [x["id"] for x in SETS["oracle"] if x["id"] in clean and clean[x["id"]]["T"] is not None]
To = np.array([clean[i]["T"] for i in orc], dtype=np.float64)
dirs, G = {}, {}
nr = {l: np.median([np.linalg.norm(Hs[f"{i}|{l}|r"]) for i in orc]) for l in LAY}
for l in LAY:
    dirs[f"o_{l}"] = oracle(np.array([Hs[f"{i}|{l}|r"] for i in orc]), To, binary=BINARY)
    dirs[f"os_{l}"] = oracle(np.array([Hs[f"{i}|{l}|s"] for i in orc]), To, binary=BINARY)
    G[l] = float(G_PUB * nr[l] / nr[ELL])  # same fraction of the residual norm as paper 6's gap at ell
gauss = gaussian(CFG["R"], m.d)
log(f"oracle split: {len(orc)} items, T mean {To.mean():.3f}" + (f", {int(To.sum())} positives" if BINARY else "") +
    "; G per layer " + " ".join(f"{l}:{G[l]:.1f}" for l in LAY) +
    "; cos(o,os) " + " ".join(f"{l}:{dirs[f'o_{l}'] @ dirs[f'os_{l}']:+.2f}" for l in LAY))
np.savez(PRE + "_dirs.npz", gauss=gauss, layers=np.array(LAY), G=np.array([G[l] for l in LAY]), **dirs)
json.dump(dict(target=TARGET, model=KEY, L=L, ell=ELL, layers=LAY, G=G, G_pub=G_PUB, n_oracle=len(orc),
               T_mean_oracle=float(To.mean()), resid_norm_median={str(l): float(nr[l]) for l in LAY},
               cfg={k: CFG[k] for k in ("k", "R", "c", "persist_offset")}),
          open(PRE + "_info.json", "w"), indent=1)

K, C = CFG["k"], CFG["c"]
rows_meta, rows_vec = [dict(kind="clean", cond="clean", l=ELL)], [(ELL, np.zeros(m.d), 0.0)]
for cond in ("a", "b", "c"):
    for l in LAY:
        for s in (1, -1):
            for k in K:
                for kind in (("o", "os") if cond == "c" else ("o",)):
                    rows_meta.append(dict(kind=kind, cond=cond, l=l, k=s * k))
                    rows_vec.append((l, dirs[f"{kind}_{l}"], s * k * G[l]))
        for j in range(CFG["R"]):
            for c in C:
                rows_meta.append(dict(kind="rand", cond=cond, l=l, j=j, c=c))
                rows_vec.append((l, gauss[j], c * G[l]))
NR = len(rows_meta)
cap_layers = list(range(min(LAY), L))
log(f"{NR} rows per item; capture layers {cap_layers[0]}..{cap_layers[-1]}")

path = PRE + "_push.jsonl"
done = done_ids(path)
D_items = SETS["D"][:N_LIMIT] if N_LIMIT else SETS["D"]
todo = [i for i in D_items if i in clean and i not in done and clean[i]["T"] is not None]
log(f"push: {len(todo)} items to run ({len(D_items)} Check D items; {len([i for i in D_items if i in clean and clean[i]['T'] is None])} without a clean target)")
byid = {x["id"]: x for x in SETS["A"] + SETS["oracle"]}
step = CFG["max_rows"] - 1
body = list(range(1, NR))
with open(path, "a", encoding="utf-8") as f:
    for n, iid in enumerate(todo):
        t_item = time.time()
        it, crec = byid[iid], clean[iid]
        ids, start, src = source_positions(m, it)
        npr = len(ids)
        cap_pos = [start, npr - 1]  # reading position, last prompt token (predicts the first answer token)
        out = [dict(rows_meta[b]) for b in range(NR)]
        pers = {}  # key -> list over cap_layers of [P1, P2] survival (o/os rows) ; rand: summed over j
        cnt = {}
        for c0 in range(0, len(body), step):
            ch = [0] + body[c0:c0 + step]
            B = len(ch)
            add = {l: torch.zeros(B, m.d) for l in LAY}
            U = torch.zeros(B, m.d)
            S = torch.zeros(B)
            pm = torch.zeros(B, npr)
            gm = torch.zeros(B)
            for bi, b in enumerate(ch):
                l, u, sc = rows_vec[b]
                if sc != 0:
                    add[l][bi] = torch.tensor(sc * u, dtype=torch.float32)
                U[bi] = torch.tensor(u, dtype=torch.float32)
                S[bi] = sc
                p_, g_ = masks(rows_meta[b]["cond"], npr, start, src)
                pm[bi], gm[bi] = p_, g_
            r = m.gen8(ids, add, pm, gm, N_NEW, lens=LENS)
            oc0 = spec.outcome(m, it, MODE, r["tokens"][0], r["lp0"][0], r["top1"][0] if LENS else None)
            if c0 == 0:
                tok0, T0c = r["tokens"][0], T_of(oc0, crec)
            elif not (torch.equal(r["tokens"][0], tok0) and T_of(oc0, crec) == T0c):
                out[0]["chunk_clean_mismatch"] = out[0].get("chunk_clean_mismatch", 0) + 1
            for bi, b in enumerate(ch):
                if c0 > 0 and bi == 0:
                    continue
                oc = spec.outcome(m, it, MODE, r["tokens"][bi], r["lp0"][bi], r["top1"][bi] if LENS else None)
                out[b]["T"] = T_of(oc, crec)
                out[b]["ok"] = oc.get("correct")
            cont = m.trunc(r["tokens"][0])
            kl, cap = m.tf8(ids, start, add, pm, gm, cont, cap_layers, cap_pos)
            # persistence: change of the projection on the written unit direction, relative to clean, in
            # units of the written amount (1 = the write survives unchanged; 0 = gone or never arrived)
            Ug = U.to(m.dev)
            den = torch.where(S == 0, torch.ones_like(S), S).to(m.dev)[:, None]
            surv = []
            for lp in cap_layers:
                Dh = cap[lp] - cap[lp][0:1]  # [B, 2, d]
                surv.append(torch.einsum("bpd,bd->bp", Dh, Ug) / den)
            surv = torch.stack(surv, 1).cpu().numpy()  # [B, len(cap_layers), 2]
            del cap
            for bi, b in enumerate(ch):
                if c0 > 0 and bi == 0:
                    continue
                out[b]["kl"] = kl[bi]
                rm = rows_meta[b]
                if rm["kind"] == "clean":
                    continue
                li = cap_layers.index(rm["l"])
                curve = surv[bi, li:]  # layers l .. L-1
                if rm["kind"] in ("o", "os"):
                    pers[f"{rm['kind']}|{rm['cond']}|{rm['l']}|{rm['k']}"] = np.round(curve, 4).tolist()
                elif rm["c"] in K:
                    key = f"rand|{rm['cond']}|{rm['l']}|{rm['c']}"
                    pers[key] = (np.asarray(pers[key]) + curve).tolist() if key in pers else curve.tolist()
                    cnt[key] = cnt.get(key, 0) + 1
        for key, v in cnt.items():
            pers[key] = np.round(np.asarray(pers[key]) / v, 4).tolist()
        f.write(json.dumps(dict(id=iid, n_prompt=npr, start=start, n_src=len(src), rows=out, persist=pers)) + "\n")
        f.flush()
        if n % 5 == 0:
            oa = [x for x in out if x["kind"] == "o" and x["l"] == ELL and x["k"] == 4]
            log(f"push {n}/{len(todo)} {iid} {time.time() - t_item:.1f}s; clean T={out[0]['T']} | o(+4,ell) " +
                " ".join(f"{x['cond']}:kl={x['kl']:.1f},T={x['T']}" for x in oa))
log("DONE")
