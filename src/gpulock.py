"""Atomic directory lock for the shared GPU (../_datathon_shared/gpu.lock). Stale after 2h."""
import os, json, time, socket, contextlib, shutil

LOCK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "_datathon_shared", "gpu.lock"))


@contextlib.contextmanager
def gpu_lock(owner="track2", stale_s=7200):
    os.makedirs(os.path.dirname(LOCK), exist_ok=True)
    got = False
    try:
        try:
            os.mkdir(LOCK); got = True
        except FileExistsError:
            meta = os.path.join(LOCK, "owner.json")
            age = time.time() - os.path.getmtime(LOCK)
            if age > stale_s:
                shutil.rmtree(LOCK, ignore_errors=True); os.mkdir(LOCK); got = True
        if got:
            json.dump({"owner": owner, "host": socket.gethostname(), "pid": os.getpid(), "t": time.time()}, open(os.path.join(LOCK, "owner.json"), "w"))
        yield got
    finally:
        if got: shutil.rmtree(LOCK, ignore_errors=True)
