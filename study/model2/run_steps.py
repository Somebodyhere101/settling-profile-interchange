"""Driver: python run_steps.py <steps.json> <tag>. Runs [cwd, [args...], log] steps in order, stopping at the
first failure; logs to logs/<log>, progress to logs/driver_<tag>.log. Asks Windows not to idle-sleep."""

import ctypes
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
steps = json.load(open(sys.argv[1], encoding="utf-8"))
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
env = dict(os.environ, PYTHONIOENCODING="utf-8", MAX_ROWS="64")
os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
try:
    for cwd, args, log in steps:
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
