"""Paper 8 lens-robustness arm (lens/PREDICTIONS.md): is settling depth a process variable or a lens artefact?

python run_lens.py <items|selftest|run> [n_items]

items     builds res/items.json: the 381 arm-S2 items plus 500 fresh correct items (all correct pool M items,
          then pool G items), none in arm S2 or in the tuned lens's training set; and for every item a
          corrupt partner (different operands, a different digit at every answer position, same token
          layout and answer length).
selftest  engine checks on 3 items outside the measured set.
run       for each item and each prompt form (original "What is a * b?", paraphrase "Calculate a * b."):
          one teacher-forced pass on the item's own answer, giving per-digit settle layers under (a) the
          logit lens and (b) the frozen tuned lens; then (c) causal arrival: the clean block-l output at the
          position predicting digit k is patched into the corrupt partner's teacher-forced run (same
          position), for every l and k; the patched run's output at that position is read.
          res/lens.jsonl
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

STAGE = sys.argv[1]
N_LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else None
KEY = "sm2-1.7b"
MAX_ROWS = int(os.environ.get("MAX_ROWS", "64"))
RES = os.path.join(HERE, "res")
os.makedirs(RES, exist_ok=True)
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
T0 = time.time()
FORMS = {"orig": "What is {a} * {b}?", "para": "Calculate {a} * {b}."}
N_FRESH = 500


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


spec = D3()
P = spec.P


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")] if os.path.exists(path) else []


def item(i, form):
    a, b = (int(v) for v in i[2:].split("*"))
    return dict(id=i, a=a, b=b, answer=a * b, prompt=FORMS[form].format(a=a, b=b), prefill=f"{a} * {b} = ")


def prompt_ids(m, it):
    return m.ids(m.chat(it["prompt"]) + it["prefill"])


m = SRModel(KEY)
L = m.L
pool = {r["id"]: r for r in read_jsonl(os.path.join(OLD, "res", "pool_clean.jsonl"))}
pairs = json.load(open(os.path.join(P2, "res", "pairs2.json"), encoding="utf-8"))["S"]
S2_ITEMS = set()
for x in pairs:
    S2_ITEMS |= {x["r"], *x["far"], *x["null"]}
ITEMS_PATH = os.path.join(RES, "items.json")


def lens_train_items():
    """Exactly the 1,500 arithmetic items of ../selfreport/run_sr.py fitlens (rng seed 5, first draw)."""
    rng = np.random.default_rng(5)
    arith = [i for i, r in sorted(pool.items()) if r["set"] == "G" and i not in S2_ITEMS and r["correct"] == 1]
    return {arith[j] for j in rng.choice(len(arith), 1500, replace=False)}


def digits_of(i):
    _, idx = P.answer_span(m.tok, pool[i]["cont"])
    return [m.tok.decode([pool[i]["cont"][k]]) for k in idx], idx


# ==================================================================== items
if STAGE == "items":
    excl = S2_ITEMS | lens_train_items()
    ok = lambda i: pool[i]["correct"] == 1 and i not in excl
    fresh_M = [i for i, r in sorted(pool.items()) if r["set"] == "M" and ok(i)]
    rng = np.random.default_rng(8)
    G = [i for i, r in sorted(pool.items()) if r["set"] == "G" and ok(i)]
    fresh_G = [G[j] for j in rng.choice(len(G), N_FRESH - len(fresh_M), replace=False)]
    measured = sorted(S2_ITEMS) + fresh_M + fresh_G
    # corrupt partners: correct pool items outside the measured set, same answer-token sequence length and
    # digit positions, a different digit at every digit position, same prompt token length in both forms
    cands = [i for i, r in sorted(pool.items()) if r["correct"] == 1 and i not in set(measured)]
    lay = {}

    def layout_key(i):
        if i not in lay:
            dg, idx = digits_of(i)
            lay[i] = (len(pool[i]["cont"]), tuple(idx), tuple(len(prompt_ids(m, item(i, f))) for f in FORMS))
        return lay[i]
    by_key = {}
    for i in cands:
        by_key.setdefault(layout_key(i), []).append(i)
    partner = {}
    for i in measured:
        dg, _ = digits_of(i)
        a, b = item(i, "orig")["a"], item(i, "orig")["b"]
        opts = [j for j in by_key.get(layout_key(i), [])
                if all(x != y for x, y in zip(digits_of(j)[0], dg)) and {item(j, "orig")["a"], item(j, "orig")["b"]} != {a, b}]
        if opts:
            partner[i] = opts[int(rng.integers(0, len(opts)))]
    measured = [i for i in measured if i in partner]
    json.dump(dict(items=measured, partner=partner, n_s2=len([i for i in measured if i in S2_ITEMS]),
                   n_fresh_M=len([i for i in measured if i in set(fresh_M)]), n_fresh_G=len([i for i in measured if i in set(fresh_G)])),
              open(ITEMS_PATH, "w"), indent=1)
    log(f"items: {len(measured)} measured (S2 {len([i for i in measured if i in S2_ITEMS])}, fresh M "
        f"{len([i for i in measured if i in set(fresh_M)])}, fresh G {len([i for i in measured if i in set(fresh_G)])}); "
        f"lens-training items excluded {len(lens_train_items())}")
    sys.exit(0)

m.tuned = torch.load(os.path.join(SR1, "res", "tuned_lens.pt"), weights_only=False)["W"].to(m.dev)
IT = json.load(open(ITEMS_PATH, encoding="utf-8"))


def measure(i, form, check=False):
    """One item, one prompt form: settle (logit, tuned) and the causal arrival table."""
    j = IT["partner"][i]
    ci_, cj = pool[i]["cont"], pool[j]["cont"]
    si, sj = prompt_ids(m, item(i, form)) + ci_, prompt_ids(m, item(j, form)) + cj
    assert len(si) == len(sj)
    n = len(si) - len(ci_)
    _, idx = P.answer_span(m.tok, ci_)
    dpos = [n - 1 + k for k in idx]
    dig_i = [ci_[k] for k in idx]
    dig_j = [cj[k] for k in idx]
    rows = [(l, kk) for kk in range(len(dpos)) for l in range(L)]
    out_top, out_ld = {}, {}
    clean = None
    step = MAX_ROWS - 2
    for c0 in range(0, len(rows), step):
        ch = rows[c0:c0 + step]
        B = 2 + len(ch)
        patch = {}
        for b, (l, kk) in enumerate(ch):
            if l not in patch:
                patch[l] = (torch.zeros(B, len(si), dtype=torch.bool), torch.arange(B))
            patch[l][0][2 + b, dpos[kk]] = True
            patch[l][1][2 + b] = 0
        h, cap = m.tf_pass([si, sj] + [sj] * len(ch), patch)
        if clean is None:
            ll = m.lens_top1(cap, dpos).cpu()
            tl = m.lens_top1(cap, dpos, tuned=True).cpu()
            clean = dict(ll=[int(P.settle_top1(ll[0, k])) for k in range(len(dpos))],
                         tl=[int(P.settle_top1(tl[0, k])) for k in range(len(dpos))],
                         argmax_ok=int(all(int(ll[0, k, -1]) == dig_i[k] for k in range(len(dpos)))))
            lg0 = m.hf.lm_head(h[0:2, dpos]).float()  # [2, ndig, V]
            clean["ld_clean"] = [float(lg0[0, k, dig_i[k]] - lg0[0, k, dig_j[k]]) for k in range(len(dpos))]
            clean["ld_corrupt"] = [float(lg0[1, k, dig_i[k]] - lg0[1, k, dig_j[k]]) for k in range(len(dpos))]
            clean["corrupt_argmax_is_own"] = int(all(int(lg0[1, k].argmax()) == dig_j[k] for k in range(len(dpos))))
            if check:
                clean["_cap"] = {l: cap[l][:2].clone() for l in range(L)}
        del cap
        for b, (l, kk) in enumerate(ch):
            lg = m.hf.lm_head(h[2 + b, dpos[kk]]).float()
            out_top[(l, kk)] = int(lg.argmax() == dig_i[kk])
            out_ld[(l, kk)] = float(lg[dig_i[kk]] - lg[dig_j[kk]])
        del h
    nd = len(dpos)
    clean["top"] = [[out_top[(l, kk)] for l in range(L)] for kk in range(nd)]
    clean["ld"] = [[round(out_ld[(l, kk)], 3) for l in range(L)] for kk in range(nd)]
    return clean


# ================================================================ selftest
if STAGE == "selftest":
    # (1) a patch at the last block reproduces the clean digit's output exactly; (2) patching at layer l leaves the
    # corrupt run unchanged before position p_k; (3) clean settle under teacher forcing equals the pool's (orig form)
    others = [i for i in pool if i not in set(IT["items"]) and i not in S2_ITEMS and pool[i]["correct"] == 1 and pool[i]["set"] == "G"][:200]
    rng = np.random.default_rng(1)
    worst = dict(last_layer_top=1, settle_match=0, n=0)
    for i in [IT["items"][-1], IT["items"][-2], IT["items"][-3]]:
        r = measure(i, "orig")
        worst["last_layer_top"] = min(worst["last_layer_top"], min(t[-1] for t in r["top"]))
        worst["settle_match"] += int(r["ll"] == pool[i]["settle"])
        worst["n"] += 1
        log(i, "ll", r["ll"], "pool", pool[i]["settle"], "tl", r["tl"], "arrival top rows", ["".join(map(str, t)) for t in r["top"]])
    log("selftest", worst)
    assert worst["last_layer_top"] == 1, "patching the last block output does not give the clean digit"
    assert worst["settle_match"] >= worst["n"] - 1, "teacher-forced settle differs from the pool"
    log("SELFTEST PASSED")
    sys.exit(0)

# ===================================================================== run
assert STAGE == "run", STAGE
path = os.path.join(RES, "lens" + ("_smoke" if N_LIMIT else "") + ".jsonl")
done = {(r["id"], r["form"]) for r in read_jsonl(path)}
todo = [(i, f) for i in (IT["items"][:N_LIMIT] if N_LIMIT else IT["items"]) for f in FORMS if (i, f) not in done]
log(f"lens: {len(todo)} (item, form) to run")
with open(path, "a", encoding="utf-8") as fh:
    for nn, (i, f) in enumerate(todo):
        r = measure(i, f)
        fh.write(json.dumps(dict(id=i, form=f, s2=int(i in S2_ITEMS), set=pool[i]["set"], partner=IT["partner"][i], **r)) + "\n")
        fh.flush()
        if nn % 50 == 0:
            log(f"lens {nn}/{len(todo)} {i} {f} ll {r['ll']} tl {r['tl']}")
log("DONE")
