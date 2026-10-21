"""Checks the abstract: plain ASCII and at most 1,900 characters. python abstract_check.py"""
import os
import sys

s = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.tex"), encoding="utf-8").read()
b, e = "\\begin{abstract}", "\\end{abstract}"
a = s[s.index(b) + len(b):s.index(e)].strip()
ok = len(a) <= 1900 and all(ord(c) < 128 for c in a)
print(f"abstract: {len(a)} characters, ascii={all(ord(c) < 128 for c in a)} -> {'OK' if ok else 'TOO LONG / NON-ASCII'}")
sys.exit(0 if ok else 1)
