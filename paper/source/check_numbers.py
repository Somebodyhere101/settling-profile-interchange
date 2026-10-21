"""Recomputes every number used in the abstract, introduction and conclusions of main.tex from the released
JSON files, and checks that each appears in main.tex with the stated rounding.

python check_numbers.py [study_dir]   (default: repo/study, i.e. the release; falls back to ../paper8)

Each claim: (label, value computed from the data, format, text that must appear in main.tex). The script prints
every claim with PASS/FAIL and exits non-zero if any claim fails or a source file is missing.
"""

import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
S = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "repo", "study")
if not os.path.exists(S):
    S = os.path.join(HERE, "..", "paper8")
TEX = open(os.path.join(HERE, "main.tex"), encoding="utf-8").read()
ZIP = os.path.join(HERE, "zenodo_upload", "results-per-item.zip")


def J(rel):
    return json.load(open(os.path.join(S, rel), encoding="utf-8"))


def jsonl(rel):
    p = os.path.join(S, rel)
    if os.path.exists(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    with zipfile.ZipFile(ZIP) as z:  # per-item files are only in the Zenodo zip
        return [json.loads(l) for l in z.read(rel.replace("\\", "/")).decode("utf-8").splitlines() if l.strip()]


f2 = lambda x: f"{abs(x):.2f}" if abs(x) < 0.005 else f"{x:.2f}"
ci2 = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"

pil = J("code/res/q3-1.7b_summary.json")
st = pil["targets"]["settle"]["cells"]
fl = pil["targets"]["flip"]["cells"]
p2 = J("patch2/res/patch2_summary.json")
k1 = J("selfreport/res/kill_summary.json")
s1 = J("selfreport/res/sr_summary.json")
s2 = J("selfreport2/res/sr2_summary.json")
ln = J("lens/res/lens_summary.json")
ip = J("inputproxy/res/ip_summary.json")
m2p = J("model2/patch2/res/patch2_summary.json")
m2k = J("model2/selfreport/res/kill_summary.json")
m2s = J("model2/selfreport2/res/sr2_summary.json")
m2l = J("model2/lens/res/lens_summary.json")
pool = {r["id"]: r for r in jsonl("patch/res/pool_clean.jsonl")}
pairs = J("patch2/res/pairs2.json")["S"]
items = sorted({x["r"] for x in pairs} | {d for x in pairs for d in x["far"] + x["null"]})
later = [s for i in items for s in pool[i]["settle"][1:]]
frac_above21 = sum(s > 21 for s in later) / len(later)

C = []


def claim(label, value, fmt, text):
    C.append((label, value, fmt(value) if callable(fmt) else fmt % value, text))


claim("pilot: oracle readability min", min(v["est"] for k, v in pil["targets"]["settle"]["readability"].items() if isinstance(v, dict)), f2, "0.47")
claim("pilot: (a)@14 4 gaps effect", st["o|a|14|4|-1"]["effect"], lambda x: f"+{x:.3f}", "+0.049")
claim("pilot: (b)@14 2 gaps effect", st["o|b|14|2|-1"]["effect"], lambda x: f"+{x:.3f}", "+0.040")
claim("pilot: (b)@14 4 gaps effect", st["o|b|14|4|-1"]["effect"], lambda x: f"+{x:.3f}", "+0.045")
claim("pilot: settle moves under (b)", pil["verdicts"]["settle_moves"]["b"], lambda b: "``moves'' depth" if b else "no", "``moves'' depth")
claim("pilot: source cells controllable", (sum(v.get("controllable", False) for k, v in st.items() if "|c|" in k), sum("|c|" in k for k in st)), lambda t: f"{t[0]} of {t[1]}", "0 of 36")
claim("pilot: flip os|c|12 2 gaps", fl["os|c|12|2|1"]["effect"], lambda x: f"+{x:.2f}", "+0.29")
claim("S2: (iv)@21 tau_gen", p2["verdict"]["held"]["21"]["tau_gen"], f2, "0.91")
claim("S2: (iv)@21 tau_gen CI", p2["verdict"]["held"]["21"]["tau_gen_ci"], ci2, "[0.81, 1.04]")
claim("S2: (iv)@14 tau_gen", p2["verdict"]["held"]["14"]["tau_gen"], f2, "0.17")
claim("S2: (ii)@7 D_gen tau", p2["cells"]["ii|7"]["D_gen"]["tau"], f2, "0.56")
claim("S2: (ii)@14 D_gen tau", p2["cells"]["ii|14"]["D_gen"]["tau"], f2, "0.39")
claim("S2: (iii)@7 D_gen tau", p2["cells"]["iii|7"]["D_gen"]["tau"], f2, "0.88")
claim("S2: (iv)@21 same answer", p2["cells"]["iv|21"]["same_tokens"]["mean"], lambda x: f"{100 * x:.0f}\\%", "98\\%")
claim("S2: later-digit settle layers above 21", frac_above21, lambda x: f"{100 * x:.0f}\\%", "99\\%")
claim("SR1: (iv)@21 tau_r", s1["cells"]["iv@21"]["r"]["tau"], f2, "0.00")
claim("SR1: (iv)@21 tau_r CI", s1["cells"]["iv@21"]["r"]["tau_ci"], ci2, "[-0.06, 0.04]")
claim("SR1: (i)@7 tau_r", s1["cells"]["i@7"]["r"]["tau"], f2, "0.70")
claim("SR1: clean rho(r, D)", s1["clean_spearman_r_D"]["rho"], f2, "-0.01")
claim("SR1: clean mean r", s1["clean_r"]["mean"], lambda x: f"{x:.1f} nats", "10.7 nats")
claim("SR2: (iv)@21 tau", s2["cells"]["iv@21"]["tau"], f2, "0.00")
claim("SR2: (iv)@21 tau CI", s2["cells"]["iv@21"]["tau_ci"], ci2, "[-0.03, 0.04]")
claim("SR2: (iv+rep)@21 tau", s2["cells"]["ivrep@21"]["tau"], f2, "0.99")
claim("SR2: (ii)@7 tau", s2["cells"]["ii@7"]["tau"], f2, "0.66")
claim("SR2: (i)@7 tau", s2["cells"]["i@7"]["tau"], f2, "0.20")
claim("SR2: input-predicted share of gap", s2["input_predicted_gap"]["tau"], f2, "0.65")
claim("SR2: (i)@7 effect minus input-predicted", s2["input_predicted_cells"]["i@7"]["effect_minus_predicted"]["effect"], lambda x: f"{x:.3f}", "-0.080")
claim("SR2: clean gap", s2["gap"]["effect"], lambda x: f"+{x:.3f}", "+0.177")
claim("kill: tuned tau_gen (primary)", k1["cells"]["tl|D_gen|iv@21"]["tau"], f2, "1.83")
claim("kill: tuned gap layers", k1["cells"]["tl|D_gen|iv@21"]["gap"] * 28, f2, "0.17")
claim("kill: copy-free tuned tau", k1["cells"]["tl|D_gen_above|iv@21"]["tau"], f2, "1.06")
claim("lens: rho(a,c)", ln["primary"]["a_c"]["rho"], f2, "0.26")
claim("lens: rho(a,c) CI", ln["primary"]["a_c"]["ci"], ci2, "[0.20, 0.32]")
claim("lens: S2 gap under arrival", ln["s2_grouping"]["c"]["gap_layers"], f2, "0.18")
claim("lens: S2 gap under arrival CI", ln["s2_grouping"]["c"]["ci"], ci2, "[-0.60, 0.83]")
claim("lens: reliability (logit) min", min(ln["reliability_orig_vs_para"][k]["rho"] for k in ("ll", "tl", "c")), f2, "0.88")
claim("input proxy: within-item reliability", ip["reliability_within"]["pearson"], f2, "0.17")
claim("input proxy: between-item variance share", 1 - ip["variance"]["within_item_mean"] / ip["variance"]["total"], lambda x: f"{100 * x:.0f}\\%", "92\\%")
claim("SmolLM2: (iv)@18 tau_gen", m2p["verdict"]["held"]["18"]["tau_gen"], f2, "0.79")
claim("SmolLM2: (iv)@12 tau_gen", m2p["verdict"]["held"]["12"]["tau_gen"], f2, "0.04")
claim("SmolLM2: lens rho(a,c)", m2l["primary"]["a_c"]["rho"], f2, "0.40")
claim("SmolLM2: self-report clean gap", m2s["gap"]["effect"], lambda x: f"+{x:.3f}", "+0.004")

claim("SR2: (i)@7 effect minus predicted CI", s2["input_predicted_cells"]["i@7"]["effect_minus_predicted"]["ci"], lambda c: f"[{c[0]:.3f}, {c[1]:.3f}]", "[-0.134, -0.036]")
claim("kill: tuned tau_gen CI", k1["cells"]["tl|D_gen|iv@21"]["tau_ci"], ci2, "[-6.75, 11.11]")
claim("kill: tuned gap CI includes 0", k1["cells"]["tl|D_gen|iv@21"]["gap_ci"][0] < 0, lambda b: "a CI that includes 0" if b else "no", "a CI that includes 0")
claim("S2: later-digit settle min layer", min(later), lambda x: f"none below {x}", "none below 20")
claim("S2: (iii+)@24 D_gen tau", p2["cells"]["iiip|24"]["D_gen"]["tau"], f2, "0.81")
claim("SmolLM2: (iii+)@21 D_gen tau", m2p["cells"]["iiip|21"]["D_gen"]["tau"], f2, "0.76")
claim("SmolLM2: (iv)@21 D_gen tau", m2p["cells"]["iv|21"]["D_gen"]["tau"], f2, "0.72")
claim("lens: S2-only rho(a,c)", ln["secondary_measures"]["a_c_S2only"]["rho"], f2, "0.19")
claim("lens: S2-only rho(b,c)", ln["secondary_measures"]["b_c_S2only"]["rho"], f2, "0.03")
claim("SmolLM2: S2 gap under arrival", m2l["s2_grouping"]["c"]["gap_layers"], f2, "0.65")
claim("SmolLM2: S2 gap under arrival CI", m2l["s2_grouping"]["c"]["ci"], ci2, "[0.13, 1.28]")
claim("SmolLM2: kill copy-free tau", m2k["cells"]["tl|D_gen_above|iv@18"]["tau"], f2, "0.79")
claim("SR2: y_self", s2["clean_report_vs_own_settle"]["mean_y_self_layers"], f2, "-0.33")

bad = 0
for label, val, shown, text in C:
    ok = shown == text and text in TEX
    bad += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {label:45s} data -> {shown:16s} text '{text}' {'found' if text in TEX else 'NOT FOUND'}")
print(f"\n{len(C) - bad}/{len(C)} claims pass (study dir: {S})")
sys.exit(1 if bad else 0)
