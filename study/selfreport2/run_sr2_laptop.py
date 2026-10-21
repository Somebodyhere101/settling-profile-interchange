"""Driver: run_sr2.py sr2, then analyze_sr2.py, while asking Windows not to idle-sleep (no setting changed)."""

import ctypes
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
env = dict(os.environ, PYTHONIOENCODING="utf-8")
try:
    for args, log in ((["run_sr2.py", "sr2"], "sr2.log"), (["analyze_sr2.py"], "analyze_sr2.log")):
        print(time.strftime("%H:%M:%S"), "start", *args, flush=True)
        with open(os.path.join(HERE, "logs", log), "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable] + args, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "end", *args, "rc", rc, flush=True)
        if rc != 0:
            print("FAILED", flush=True)
            sys.exit(rc)
    print("ALL DONE", flush=True)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
