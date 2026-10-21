"""Paper 8 input-proxy arm (inputproxy/PREDICTIONS.md): is the readable "process" an input correlate?

python run_ip.py <items|selftest|run>

items     150 base items (correct 4-digit G items outside arm S2, the tuned lens's training set and the lens arm's
          measured set; seed 11) x 7 context variants; greedy answers; keeps a variant only if its answer tokens
          equal the base variant's (v0). Picks 4 fixed corrupt items whose digits differ pairwise at every
          position. res/items.json (no target or probe value is computed).
selftest  checks on 2 items outside the base set: the norm-rescaled replacement lands as specified; a last-block
          replacement reproduces the clean digit; a variant's corrupt rows are identical whichever item they serve.
run       per (item, variant): reading-position and last-prompt-token block outputs at layers 14 and 21 (probe
          features); per digit, causal arrival: the clean block-l output at the digit's predicting position,
          rescaled to the norm of the corrupt run's own block-l output at its digit position, replaces that
          output in a fixed corrupt run (teacher-forced, never otherwise intervened); measured with corrupt run A
          (the first fixed corrupt item whose digit differs) and B (the second). Also logit-lens settle layers.
          res/ip.jsonl, res/ip_h.npz
"""

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SR1 = os.path.join(HERE, "..", "selfreport")
OLD = os.path.join(HERE, "..", "patch")
sys.path.insert(0, SR1)
sys.path.insert(0, os.path.join(HERE, "..", "code"))

from srlib import SRModel  # noqa: E402
from specs import D3  # noqa: E402

STAGE = sys.argv[1]
KEY = "q3-1.7b"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
T0 = time.time()
PROBE_LAYERS = [14, 21]
N_BASE = 150
VARIANTS = {
    "v0": (None, "What is {a} * {b}?"),
    "v1": (None, "What is 12 * 34? 408\nWhat is 56 * 78? 4368\nWhat is {a} * {b}?"),
    "v2": (None, "Please read the following question carefully and answer it. What is {a} * {b}?"),
    "v3": (None, "Compute the product of {a} and {b}."),
    "v4": (None, "Note that 17 + 25 = 42. What is {a} * {b}?"),
    "v5": ("You are a precise calculator.", "What is {a} * {b}?"),
    "v6": (None, "Calculate {a} * {b}."),
}


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = D3()
P = spec.P
m = SRModel(KEY)
L, d = m.L, m.d


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


pool = {r["id"]: r for r in read_jsonl(os.path.join(OLD, "res", "pool_clean.jsonl"))}
ITEMS_PATH = os.path.join(RES, "items.json")


def ab(i):
    return tuple(int(v) for v in i[2:].split("*"))


def texts(i, v):
    a, b = ab(i)
    sysm, user = VARIANTS[v]
    chat = m.chat(user.format(a=a, b=b), system=sysm)
    k = chat.rindex("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    return chat, chat[:k], f"{a} * {b} = "


def ids_of(i, v):
    chat, head, pre = texts(i, v)
    ids = m.ids(chat + pre)
    rp = len(m.ids(head)) - 1  # reading position: the "\n" after "assistant" (paper 6 / 8)
    assert m.ids(head) == ids[:rp + 1]
    return ids, rp


# ===================================================================== items
if STAGE == "items":
    pairs = json.load(open(os.path.join(HERE, "..", "patch2", "res", "pairs2.json"), encoding="utf-8"))["S"]
    s2 = set()
    for x in pairs:
        s2 |= {x["r"], *x["far"], *x["null"]}
    rng5 = np.random.default_rng(5)
    arith = [i for i, r in sorted(pool.items()) if r["set"] == "G" and i not in s2 and r["correct"] == 1]
    lens_train = {arith[j] for j in rng5.choice(len(arith), 1500, replace=False)}
    lens_meas = set(json.load(open(os.path.join(HERE, "..", "lens", "res", "items.json"), encoding="utf-8"))["items"])
    excl = s2 | lens_train | lens_meas
    cand = [i for i, r in sorted(pool.items()) if r["set"] == "G" and r["correct"] == 1 and len(r["cont"]) == 4 and i not in excl]
    rng = np.random.default_rng(11)
    order = [cand[j] for j in rng.permutation(len(cand))]
    # 4 fixed corrupt items: digits pairwise different at every position (so >= 3 differ from any clean digit)
    corrupt = []
    for i in order:
        dg = pool[i]["cont"]
        if all(all(dg[k] != pool[c]["cont"][k] for k in range(4)) for c in corrupt):
            corrupt.append(i)
        if len(corrupt) == 4:
            break
    base = [i for i in order if i not in corrupt][:N_BASE]
    keep = {}
    n_tried = n_kept = 0
    for i in base:
        keep[i] = []
        for v in VARIANTS:
            ids, rp = ids_of(i, v)
            g = m.gen([ids], None, 8, lens=False)
            tt = m.trunc(g["tokens"][0])
            _, idx = P.answer_span(m.tok, tt)
            cont = [int(t) for t in tt[:idx[-1] + 1]] if idx else []
            n_tried += 1
            if cont == pool[i]["cont"]:
                keep[i].append(v)
                n_kept += 1
    json.dump(dict(base=base, corrupt=corrupt, keep=keep, variants=VARIANTS), open(ITEMS_PATH, "w"), indent=1)
    log(f"items: {len(base)} base x {len(VARIANTS)} variants; kept {n_kept}/{n_tried} (answer identical to v0's); "
        f"per variant {dict((v, sum(v in k for k in keep.values())) for v in VARIANTS)}; base items with >= 4 variants "
        f"{sum(len(k) >= 4 for k in keep.values())}; corrupt {corrupt}")
    sys.exit(0)

IT = json.load(open(ITEMS_PATH, encoding="utf-8"))
CORR = IT["corrupt"]
CIDS = {c: m.ids(texts(c, "v0")[0] + texts(c, "v0")[2]) + pool[c]["cont"] for c in CORR}
assert len({len(x) for x in CIDS.values()}) == 1
NC = len(next(iter(CIDS.values()))) - 4  # corrupt prompt length; its digit k is predicted at NC - 1 + k


@torch.no_grad()
def measure(i, v):
    ids, rp = ids_of(i, v)
    cont = pool[i]["cont"]
    seq = ids + cont
    n = len(ids)
    dpos = [n - 1 + k for k in range(4)]
    _, cap = m.tf_pass([seq])
    feats = {f"{tag}{l}": cap[l][0, p].float().cpu().numpy() for l in PROBE_LAYERS for tag, p in (("r", rp), ("e", n - 1))}
    ll = m.lens_top1(cap, dpos).cpu()
    settle = [int(P.settle_top1(ll[0, k])) for k in range(4)]
    clean_h = {(l, k): cap[l][0, dpos[k]].float() for l in range(L) for k in range(4)}
    del cap
    # corrupt runs A and B per digit
    which = {}
    for k in range(4):
        diff = [c for c in CORR if pool[c]["cont"][k] != cont[k]]
        which[k] = (diff[0], diff[1])
    _, ccap = m.tf_pass([CIDS[c] for c in CORR])
    cnorm = {(ci, l, k): float(ccap[l][ci, NC - 1 + k].float().norm()) for ci in range(len(CORR)) for l in range(L) for k in range(4)}
    cstate = {(ci, l, k): ccap[l][ci, NC - 1 + k].float() for ci in range(len(CORR)) for l in range(L) for k in range(4)}
    del ccap
    rows = [(ab_, k, l) for ab_ in (0, 1) for k in range(4) for l in range(L)]
    top = {}
    for c0 in range(0, len(rows), MAX_ROWS):
        ch = rows[c0:c0 + MAX_ROWS]
        B = len(ch)
        T = len(CIDS[CORR[0]])
        add = {l: torch.zeros(B, d) for l in range(L)}
        pm = torch.zeros(B, T)
        seqs = []
        for b, (ab_, k, l) in enumerate(ch):
            c = which[k][ab_]
            ci = CORR.index(c)
            h = clean_h[(l, k)]
            h = h * (cnorm[(ci, l, k)] / float(h.norm()))  # rescaled to the corrupt run's norm
            add[l][b] = (h - cstate[(ci, l, k)]).cpu()
            pm[b, NC - 1 + k] = 1
            seqs.append(CIDS[c])
        hh, _ = m.tf_pass(seqs, None, {l: v for l, v in add.items() if v.abs().sum() > 0}, pm)
        for b, (ab_, k, l) in enumerate(ch):
            lg = m.hf.lm_head(hh[b, NC - 1 + k]).float()
            top[(ab_, k, l)] = int(int(lg.argmax()) == cont[k])
        del hh
    arr = {}
    for ab_ in (0, 1):
        out = []
        for k in range(4):
            fl = [top[(ab_, k, l)] for l in range(L)]
            j = L
            while j > 0 and fl[j - 1]:
                j -= 1
            out.append(j)
        arr["AB"[ab_]] = out
    return feats, settle, arr


# ================================================================ selftest
if STAGE == "selftest":
    others = [i for i, r in sorted(pool.items()) if r["set"] == "G" and r["correct"] == 1 and len(r["cont"]) == 4
              and i not in set(IT["base"]) and i not in set(CORR)][:2]
    for i in others:
        feats, settle, arr = measure(i, "v0")
        log(i, "settle", settle, "arrival A", arr["A"], "B", arr["B"], "pool settle", pool[i]["settle"])
        assert max(arr["A"]) < L and max(arr["B"]) < L, "last-block replacement did not give the clean digit"
    log("SELFTEST PASSED")
    sys.exit(0)

# ===================================================================== run
assert STAGE == "run", STAGE
path = os.path.join(RES, "ip.jsonl")
hpath = os.path.join(RES, "ip_h.npz")
done = {(r["id"], r["v"]) for r in read_jsonl(path)}
H = dict(np.load(hpath)) if os.path.exists(hpath) else {}
todo = [(i, v) for i in IT["base"] for v in IT["keep"][i] if (i, v) not in done]
log(f"run: {len(todo)} (item, variant)")
pending = []
with open(path, "a", encoding="utf-8") as f:
    for nn, (i, v) in enumerate(todo):
        feats, settle, arr = measure(i, v)
        for kf, x in feats.items():
            H[f"{i}|{v}|{kf}"] = x.astype(np.float32)
        pending.append(json.dumps(dict(id=i, v=v, settle=settle, arrA=arr["A"], arrB=arr["B"])) + "\n")
        if nn % 25 == 0 or nn == len(todo) - 1:
            np.savez(hpath, **H)
            for line in pending:
                f.write(line)
            f.flush()
            pending.clear()
            log(f"run {nn}/{len(todo)} {i} {v} settle {settle} arrA {arr['A']}")
log("DONE")
