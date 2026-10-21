"""Waits until the model-2 stage-B run has finished (one GPU), then runs this arm's pre-freeze stages: items, selftest."""
import ctypes, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "..", "model2", "logs", "driver_B.log")
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
env = dict(os.environ, PYTHONIOENCODING="utf-8", MAX_ROWS="64")
try:
    while not (os.path.exists(B) and any(k in open(B).read() for k in ("ALL DONE", "FAILED"))):
        time.sleep(20)
    for args, log in ((["run_ip.py", "items"], "items.log"), (["run_ip.py", "selftest"], "selftest.log")):
        print(time.strftime("%H:%M:%S"), "start", *args, flush=True)
        with open(os.path.join(HERE, "logs", log), "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable] + args, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), "end", *args, "rc", rc, flush=True)
        if rc:
            print("FAILED", flush=True); sys.exit(rc)
    print("ALL DONE", flush=True)
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
