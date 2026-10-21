"""Builds the model-2 (SmolLM2-1.7B-Instruct) replication tree from paper 8's frozen scripts.

The tree mirrors paper8/ (code, patch, patch2, selfreport, selfreport2, lens), so every relative path in
the copied scripts resolves inside model2/. Libraries are copied unchanged, except p6lib.py, which gets one
MODELS entry. Each script copy changes only constants (model key, layers, chat-turn text). Every
replacement is asserted to match the frozen text the stated number of times, so the build stops if a
target string has moved.

Layer map by relative depth (Qwen3-1.7B 28 blocks -> SmolLM2-1.7B 24 blocks): 7->6, 14->12, 21->18, 24->21.
"""

import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..")
KEY = "sm2-1.7b"


def copy(rel, subs=(), dst=None):
    s = open(os.path.join(SRC, rel), encoding="utf-8").read()
    for old, new, n in subs:
        c = s.count(old)
        assert c == n, f"{rel}: {old!r} found {c} times, expected {n}"
        s = s.replace(old, new)
    out = os.path.join(HERE, dst or rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8", newline="\n").write(s)


def raw(rel):
    out = os.path.join(HERE, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(os.path.join(SRC, rel), out)


KEYSUB = ('KEY = "q3-1.7b"', f'KEY = "{KEY}"', 1)

# ---- libraries
copy("code/p6lib.py", [('    "q3-14b": "Qwen/Qwen3-14B",\n', '    "q3-14b": "Qwen/Qwen3-14B",\n    "sm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",\n', 1)])
for f in ("code/p8lib.py", "code/specs.py", "code/spec_d8.py", "code/cr4_parsers.py", "code/data/items.json",
          "code/data/items_d8.json", "code/data/calibration.json", "patch/patchlib.py", "patch2/patchlib2.py",
          "selfreport/srlib.py"):
    raw(f)

# ---- pool and pairing (patching arm's pool/pair stages; its patch stages are not used)
copy("patch/run_patch.py", [KEYSUB, ("LAYERS = [7, 14, 21, 24]", "LAYERS = [6, 12, 18, 21]", 1)])

# ---- arm S2 replication
copy("patch2/run_patch2.py", [KEYSUB, ("LAYERS = [7, 14, 21, 24]", "LAYERS = [6, 12, 18, 21]", 1)])
copy("patch2/analyze_patch2.py", [
    ("L = 28", "L = 24", 1),
    ("LAYERS = [7, 14, 21, 24]", "LAYERS = [6, 12, 18, 21]", 1),
    ("PRIMARY = [14, 21]", "PRIMARY = [12, 18]", 1),
    ('V["PC1"] = all(pc1[l]["passed"] for l in (7, 14, 21))', 'V["PC1"] = all(pc1[l]["passed"] for l in (6, 12, 18))', 1),
    ('V["PC1_24"] = pc1[24]["passed"]', 'V["PC1_24"] = pc1[21]["passed"]', 1),
    ("gaps[('D', 7)]", "gaps[('D', 6)]", 1),
    ("gaps[('D_gen', 7)]", "gaps[('D_gen', 6)]", 1),
])

# ---- kill check and tuned lens (self-report arm 1's stages fitlens / selftest / kill)
copy("selfreport/run_sr.py", [
    KEYSUB,
    ('KILL_CELLS = [("iv", 21), ("iv", 14), ("iiip", 21)]', 'KILL_CELLS = [("iv", 18), ("iv", 12), ("iiip", 18)]', 1),
    ('SR_CELLS = [("iv", 21), ("i", 7), ("iir", 21), ("iir", 7), ("ii", 7)]', 'SR_CELLS = [("iv", 18), ("i", 6), ("iir", 18), ("iir", 6), ("ii", 6)]', 1),
    ("GAUSS_R, GAUSS_C, GAUSS_L = 6, [0.125, 0.25, 0.5, 1, 2, 4, 8], 21", "GAUSS_R, GAUSS_C, GAUSS_L = 6, [0.125, 0.25, 0.5, 1, 2, 4, 8], 18", 1),
])
copy("selfreport/analyze_sr.py", [
    ("L = 28", "L = 24", 1),
    ('for cond, l in (("iv", 21), ("iv", 14), ("iiip", 21)):', 'for cond, l in (("iv", 18), ("iv", 12), ("iiip", 18)):', 1),
    ('x["cond"] == "iv" and x["l"] == 21]', 'x["cond"] == "iv" and x["l"] == 18]', 1),
    ('out["cells"]["tl|D_gen|iv@21"]', 'out["cells"]["tl|D_gen|iv@18"]', 1),
])

# ---- self-report arm 2 design
copy("selfreport2/run_sr2.py", [
    KEYSUB,
    ('CELLS = [("iv", 21), ("ivrep", 21), ("i", 7), ("iir", 21), ("ii", 7)]', 'CELLS = [("iv", 18), ("ivrep", 18), ("i", 6), ("iir", 18), ("ii", 6)]', 1),
    ("GAUSS_R, GAUSS_C, GAUSS_L = 4, [0.125, 0.25, 0.5, 1, 2, 4, 8], 21", "GAUSS_R, GAUSS_C, GAUSS_L = 4, [0.125, 0.25, 0.5, 1, 2, 4, 8], 18", 1),
    ('FOOT = "<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n" + PREFILL', 'FOOT = "<|im_end|>\\n<|im_start|>assistant\\n" + PREFILL', 1),
    ("for k in range(22)))", "for k in range(19)))", 1),
    ("cap[21][br, rep]", "cap[18][br, rep]", 1),
    ("cap[21][1, rep]", "cap[18][1, rep]", 1),
])
copy("selfreport2/analyze_sr2.py", [
    ("L = 28", "L = 24", 1),
    ('CELLS = ["iv@21", "ivrep@21", "i@7", "iir@21", "ii@7"]', 'CELLS = ["iv@18", "ivrep@18", "i@6", "iir@18", "ii@6"]', 1),
    ('("iv", 21, dd)', '("iv", 18, dd)', 2),
    ('out["depth"]["iv@21"]', 'out["depth"]["iv@18"]', 1),
    ('("iv@21", "ivrep@21", "i@7", "ii@7")', '("iv@18", "ivrep@18", "i@6", "ii@6")', 1),
    ('out["input_predicted_cells"]["iv@21"]', 'out["input_predicted_cells"]["iv@18"]', 1),
    ('out["cells"]["iir@21"]', 'out["cells"]["iir@18"]', 1),
    ('cell in ("i@7", "ii@7")', 'cell in ("i@6", "ii@6")', 1),
])

# ---- lens-robustness arm
copy("lens/run_lens.py", [KEYSUB])
copy("lens/analyze_lens.py", [("L = 28", "L = 24", 1)])
print("model2 tree built")
