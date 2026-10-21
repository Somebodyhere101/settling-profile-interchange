"""Final release build for paper 8. Refuses to run while any placeholder remains.

python finalize.py

1. Stops if main.tex or any release file still contains AUTHOR-ROLE-TODO (or any other *-TODO placeholder, or a
   red \\todo{...} marker). The author must replace each with plain text first: the AUTHOR-ROLE-TODO in the
   author's own words.
2. Compiles main.pdf (pdflatex, bibtex, pdflatex x2) and stops on errors, undefined references or overfull boxes.
3. Checks the abstract (<= 1,900 characters, ASCII).
4. Rebuilds repo/ and zenodo_upload/ (build_release.py), then recomputes every headline number from the
   released JSONs (check_numbers.py repo/study).
5. Scans the release (files and both zips) for strings that must not ship.
It never publishes, pushes, or creates any repository or Zenodo record.
"""

import os
import re
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PLACEHOLDER = re.compile(r"[A-Z][A-Z-]*-TODO|\\todo\{")
CHECK_FILES = ["main.tex", "references.bib"] + [os.path.join("release_files", f) for f in os.listdir(os.path.join(HERE, "release_files"))]
# split literals, so that this file does not itself match the scan
# the network address is matched in full: its first two octets also occur as digits of KL values (e.g. 100.85046)
FORBIDDEN = ["100." + "85." + "64", "dq" + "_key", "ssh" + "-access", "App" + "Data", "scratch" + "-workspaces", "se" + "aled", "ca" + "abm"]


def run(cmd):
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, errors="replace")
    return r.returncode, r.stdout + r.stderr


def main():
    left = []
    for f in CHECK_FILES:
        for i, line in enumerate(open(os.path.join(HERE, f), encoding="utf-8", errors="replace"), 1):
            for m in PLACEHOLDER.finditer(line):
                left.append(f"{f}:{i}: {m.group(0)}")
    if left:
        print("REFUSING TO FINALIZE: placeholders remain (replace each with plain text first):")
        print("\n".join(left))
        sys.exit(2)
    for cmd in (["pdflatex", "-interaction=nonstopmode", "main.tex"], ["bibtex", "main"],
                ["pdflatex", "-interaction=nonstopmode", "main.tex"], ["pdflatex", "-interaction=nonstopmode", "main.tex"]):
        rc, out = run(cmd)
        if rc != 0 and cmd[0] == "pdflatex":
            sys.exit(f"pdflatex failed:\n{out[-2000:]}")
    log = open(os.path.join(HERE, "main.log"), encoding="utf-8", errors="replace").read()
    bad = [l for l in log.splitlines() if l.startswith("!") or "undefined" in l or "Overfull" in l]
    if bad:
        sys.exit("LaTeX problems:\n" + "\n".join(bad[:20]))
    for script, args in (("abstract_check.py", []), ("build_release.py", []), ("check_numbers.py", ["repo/study"])):
        rc, out = run([sys.executable, script] + args)
        print(out.strip().splitlines()[-1] if out.strip() else script)
        if rc != 0:
            sys.exit(f"{script} failed:\n{out[-3000:]}")
    hits = []
    for root, dirs, files in os.walk(os.path.join(HERE, "repo")):
        dirs[:] = [x for x in dirs if x != ".git"]
        for fn in files:
            p = os.path.join(root, fn)
            data = open(p, "rb").read().decode("utf-8", errors="replace")
            hits += [f"{os.path.relpath(p, HERE)}: {s}" for s in FORBIDDEN if s in data]
    for z in os.listdir(os.path.join(HERE, "zenodo_upload")):
        if z.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(HERE, "zenodo_upload", z)) as zz:
                for n in zz.namelist():
                    data = zz.read(n).decode("utf-8", errors="replace")
                    hits += [f"{z}:{n}: {s}" for s in FORBIDDEN if s in data]
    if hits:
        sys.exit("Forbidden strings in the release:\n" + "\n".join(hits[:50]))
    print("FINALIZED: repo/ and zenodo_upload/ are ready for the author to upload. Nothing was published.")


if __name__ == "__main__":
    main()
