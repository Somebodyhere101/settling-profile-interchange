"""Driver for model-2 stage A (PREDICTIONS_A.md): waits until the lens arm's laptop run has finished (one GPU),
then runs pool, pair and pair2, stopping at the first failure. Asks Windows not to idle-sleep (no setting changed)."""

import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LENS_DRIVER = os.path.join(HERE, "..", "lens", "logs", "driver.log")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
env = dict(os.environ, PYTHONIOENCODING="utf-8", MAX_ROWS="64")
try:
    while not (os.path.exists(LENS_DRIVER) and any(k in open(LENS_DRIVER).read() for k in ("ALL DONE", "FAILED"))):
        time.sleep(20)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    for cwd, args, log in (("patch", ["run_patch.py", "pool"], "pool.log"), ("patch", ["run_patch.py", "pair"], "pair.log"),
                           ("patch2", ["run_patch2.py", "pair2"], "pair2.log")):
        print(time.strftime("%H:%M:%S"), "start", cwd, *args, flush=True)
        with open(os.path.join(HERE, "logs", log), "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable] + args, cwd=os.path.join(HERE, cwd), env=env, stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "end", cwd, *args, "rc", rc, flush=True)
        if rc != 0:
            print("FAILED", flush=True)
            sys.exit(rc)
    print("ALL DONE", flush=True)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
