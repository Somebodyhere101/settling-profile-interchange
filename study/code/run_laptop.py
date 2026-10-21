"""Driver for the laptop run of the paper 8 pilot (DEVIATIONS.md 2026-10-19): the jobs_p8.txt chain for
Qwen3-1.7B, run in order, stopping at the first failure. While it runs it asks Windows not to idle-sleep
(SetThreadExecutionState, released on exit; no setting is changed).

python run_laptop.py   (uses this interpreter; logs in logs/)
"""

import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [("settle", "selftest"), ("settle", "clean"), ("settle", "push"),
         ("flip", "selftest"), ("flip", "clean"), ("flip", "push")]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
env = dict(os.environ, MAX_ROWS="64", P6_MEM_FRACTION="0.92", PYTHONIOENCODING="utf-8")
try:
    for target, stage in STEPS:
        log = os.path.join(HERE, "logs", f"{target}_q3-1.7b_{stage}.log")
        t = time.time()
        print(f"{time.strftime('%H:%M:%S')} start {target} {stage}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable, "run8.py", target, "q3-1.7b", stage], cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} end {target} {stage} rc={rc} ({time.time() - t:.0f}s)", flush=True)
        if rc != 0:
            print("FAILED", flush=True)
            sys.exit(rc)
    with open(os.path.join(HERE, "logs", "analyze_q3-1.7b.log"), "w", encoding="utf-8") as f:
        rc = subprocess.call([sys.executable, "analyze8.py", "q3-1.7b"], cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
    print(f"{time.strftime('%H:%M:%S')} analysis rc={rc}", flush=True)
    print("ALL DONE" if rc == 0 else "FAILED", flush=True)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
