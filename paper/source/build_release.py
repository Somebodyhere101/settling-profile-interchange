"""Assembles paper 8's release from the study folder (../paper8): the git repository (repo/) and the Zenodo
files (zenodo_upload/). Standard library only. Run from paper8_ms/: python build_release.py

- repo/study/ mirrors paper8/'s layout. Code, preregistrations (PREDICTIONS*.md), their hash files, deviation
  logs, results notes and item/pair id lists are copied byte for byte, so `sha256sum -c` on each arm's hash
  file runs from that arm's folder (except for the files listed in DATA.md as withheld).
- Withheld: calibration.json and items.json (they contain data from repositories without a licence; DATA.md
  says how to rebuild them), hidden-state arrays (.npz) and the fitted tuned lenses (.pt; rebuilt by
  `run_sr.py fitlens`).
- Results: every JSON/JSONL result file is stripped of model-written or prompt text. Listed keys are dropped
  and any other string longer than 40 characters is removed. Counts go to results/strip_report.json, and the
  build stops if such a string survives. Summaries go to repo/study/<arm>/res/; per-item files go only into
  zenodo_upload/results-per-item.zip.
"""

import collections
import hashlib
import json
import os
import shutil
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.normpath(os.path.join(HERE, "..", "paper8"))
REPO = os.path.join(HERE, "repo")
ZEN = os.path.join(HERE, "zenodo_upload")
NAME = "settling-profile-interchange"
MAXLEN = 40
DROP_KEYS = {"text", "texts", "prompt", "question", "options", "completion", "answer_text"}
EXEMPT_KEYS = {"verdict", "label", "label_verdict", "answer_only_predictor", "variants"}  # our analysis text and prompt templates, not model output
WITHHELD_FILES = {"calibration.json", "items.json"}
WITHHELD_NAMES = {"DEEPER.md", "p8.ps1", "items_d8.json"}  # AI planning notes; desktop deployment script; ARC items (rebuildable)
WITHHELD_EXT = {".npz", ".pt"}
SKIP_DIRS = {"__pycache__", "logs", ".claude"}
SKIP_FILES = {"files_map.json"}
strip_counts = collections.Counter()


def is_result(rel):
    return "/res/" in "/" + rel.replace("\\", "/") + "/" or rel.replace("\\", "/").startswith("res/")


def strip(x, path=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if k in EXEMPT_KEYS:
                out[k] = v
                continue
            if k in DROP_KEYS and isinstance(v, (str, list)):
                strip_counts[f"{path}/{k} (dropped key)"] += 1
                continue
            out[k] = strip(v, f"{path}/{k}")
        return out
    if isinstance(x, list):
        return [strip(v, path + "[]") for v in x]
    if isinstance(x, str) and len(x) > MAXLEN:
        strip_counts[f"{path} (long string)"] += 1
        return None
    return x


def check_clean(x):
    if isinstance(x, dict):
        return all(check_clean(v) for k, v in x.items() if k not in EXEMPT_KEYS)
    if isinstance(x, list):
        return all(check_clean(v) for v in x)
    return not (isinstance(x, str) and len(x) > MAXLEN)


def strip_file(src):
    if src.endswith(".jsonl"):
        lines = []
        for l in open(src, encoding="utf-8"):
            if l.strip():
                o = strip(json.loads(l), os.path.basename(src))
                assert check_clean(o)
                lines.append(json.dumps(o))
        return ("\n".join(lines) + "\n").encode("utf-8")
    o = strip(json.load(open(src, encoding="utf-8")), os.path.basename(src))
    assert check_clean(o)
    return json.dumps(o, indent=1).encode("utf-8")


def main():
    for d in (REPO, ZEN):
        os.makedirs(d, exist_ok=True)
        for name in os.listdir(d):  # clear everything except the git repository itself (repo/.git)
            if name == ".git":
                continue
            p = os.path.join(d, name)
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)  # a directory held open elsewhere stays, emptied of files
            else:
                os.remove(p)
        left = [os.path.join(r, f) for r, ds, fs in os.walk(d) for f in fs if ".git" not in os.path.relpath(r, d).split(os.sep)]
        assert not left, f"could not clear {d}: {left[:3]}"
    per_item = []
    withheld = []
    hashed = set()  # files named in any PREDICTIONS*.sha256: copied byte for byte (after a text check)
    for root, dirs, files in os.walk(STUDY):
        for fn in files:
            if fn.startswith("PREDICTIONS") and fn.endswith(".sha256"):
                for line in open(os.path.join(root, fn), encoding="utf-8"):
                    if line.startswith("#") or not line.strip():
                        continue
                    rel = line.split(None, 1)[1].strip().lstrip("*")
                    hashed.add(os.path.normpath(os.path.join(root, rel)))
    for root, dirs, files in os.walk(STUDY):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            src = os.path.join(root, fn)
            rel = os.path.relpath(src, STUDY)
            ext = os.path.splitext(fn)[1]
            if fn in SKIP_FILES:
                continue
            if (fn in WITHHELD_FILES and "data" in rel.replace("\\", "/").split("/")) or ext in WITHHELD_EXT or fn in WITHHELD_NAMES:
                withheld.append(rel)
                continue
            dst = os.path.join(REPO, "study", rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.normpath(src) in hashed and is_result(rel) and ext in (".json", ".jsonl"):
                raw_ok = (check_clean([json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]) if ext == ".jsonl"
                          else check_clean(json.load(open(src, encoding="utf-8"))))
                assert raw_ok, f"hashed file holds long text: {rel}"
                shutil.copyfile(src, dst)  # byte for byte, so the freeze hash verifies
                continue
            if is_result(rel) and ext in (".json", ".jsonl"):
                data = strip_file(src)
                if ext == ".jsonl" or os.path.getsize(src) > 2_000_000:
                    per_item.append((rel, data))  # per-item records: Zenodo only
                else:
                    open(dst, "wb").write(data)  # summaries, item/pair lists
            elif is_result(rel):
                continue  # other result formats (none expected)
            else:
                shutil.copyfile(src, dst)
    # paper
    os.makedirs(os.path.join(REPO, "paper", "source"), exist_ok=True)
    shutil.copyfile(os.path.join(HERE, "main.pdf"), os.path.join(REPO, "paper", f"{NAME}.pdf"))
    for f in ("main.tex", "references.bib", "build_release.py", "check_numbers.py", "abstract_check.py", "finalize.py", "fix_results_text.py"):
        shutil.copyfile(os.path.join(HERE, f), os.path.join(REPO, "paper", "source", f))
    shutil.copyfile(os.path.join(HERE, "V2_CHANGES.md"), os.path.join(REPO, "V2_CHANGES.md"))
    for f in ("README.md", "DATA.md", "LICENSE", "CITATION.cff", "requirements.txt"):
        shutil.copyfile(os.path.join(HERE, "release_files", f), os.path.join(REPO, f))
    open(os.path.join(REPO, ".gitattributes"), "w", newline="\n").write(
        "# Every file is kept byte for byte: preregistrations and code are hashed in each arm's PREDICTIONS*.sha256.\n* -text\n")
    json.dump(dict(maxlen=MAXLEN, drop_keys=sorted(DROP_KEYS), exempt_keys=sorted(EXEMPT_KEYS), counts=dict(strip_counts), withheld=sorted(withheld)),
              open(os.path.join(REPO, "study", "strip_report.json"), "w"), indent=1)
    # Zenodo files
    shutil.copyfile(os.path.join(HERE, "main.pdf"), os.path.join(ZEN, f"{NAME}.pdf"))
    with zipfile.ZipFile(os.path.join(ZEN, f"{NAME}-source.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [x for x in dirs if x != ".git"]  # the git metadata is not part of the source zip
            for fn in files:
                p = os.path.join(root, fn)
                z.write(p, os.path.relpath(p, REPO))
    with zipfile.ZipFile(os.path.join(ZEN, "results-per-item.zip"), "w", zipfile.ZIP_DEFLATED) as z:
        for rel, data in per_item:
            z.writestr(rel.replace("\\", "/"), data)
        z.writestr("strip_report.json", json.dumps(dict(counts=dict(strip_counts)), indent=1))
    sums = []
    for fn in sorted(os.listdir(ZEN)):
        h = hashlib.md5(open(os.path.join(ZEN, fn), "rb").read()).hexdigest()
        sums.append(f"{h}  {fn}  {os.path.getsize(os.path.join(ZEN, fn))}")
    open(os.path.join(ZEN, "MD5SUMS.txt"), "w").write("\n".join(sums) + "\n")
    print("repo files:", sum(len(f) for _, _, f in os.walk(REPO)), "| per-item files:", len(per_item), "| withheld:", len(withheld))
    print("strip counts:", dict(strip_counts))
    print("\n".join(sums))


if __name__ == "__main__":
    main()
