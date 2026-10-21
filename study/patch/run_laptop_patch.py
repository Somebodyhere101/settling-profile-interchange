"""Driver for the paper 8 patching arm on the laptop GPU: patchP, patchS, then the analysis, stopping at
the first failure. Keeps Windows from idle-sleeping while it runs (SetThreadExecutionState, released on
exit). The selftest, pool and pair stages ran before the freeze (PREDICTIONS.md §7).

python run_laptop_patch.py   (logs in logs/)
"""

import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [["run_patch.py", "patchP"], ["run_patch.py", "patchS"], ["analyze_patch.py"]]
ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
env = dict(os.environ, MAX_ROWS="64", P6_MEM_FRACTION="0.92", PYTHONIOENCODING="utf-8")
try:
    for st in STEPS:
        name = st[-1].replace(".py", "")
        t = time.time()
        print(f"{time.strftime('%H:%M:%S')} start {name}", flush=True)
        with open(os.path.join(HERE, "logs", f"{name}.log"), "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable] + st, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(f"{time.strftime('%H:%M:%S')} end {name} rc={rc} ({time.time() - t:.0f}s)", flush=True)
        if rc != 0:
            print("FAILED", flush=True)
            sys.exit(rc)
    print("ALL DONE", flush=True)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
