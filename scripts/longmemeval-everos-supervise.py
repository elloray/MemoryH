"""Supervise the existing single 500-question run; resume from validated checkpoints."""

import fcntl
import subprocess
import sys
import time

from longmemeval_everos_common import OUT, ROOT, put

STAGES = {
    "retrieve": (
        "retrieval.lock",
        "retrieval",
        ROOT / "scripts/longmemeval-everos-retrieve.py",
    ),
    "qa": ("qa.lock", "judgments", ROOT / "scripts/longmemeval-everos-grade.py"),
}


def running(lock_name):
    with (OUT / lock_name).open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def log_tail(name):
    path = OUT / f"{name}.log"
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(max(0, f.seek(0, 2) - 1500))
        return f.read().decode(errors="replace").strip().splitlines()


def restartable(tail):
    if not tail or not any(
        "Traceback (" in line or "Error:" in line or "Exception:" in line
        for line in tail
    ):
        return True
    last = tail[-1].lower()
    return any(
        s in last
        for s in (
            "timed out",
            "timeouterror",
            "connection reset",
            "connection refused",
            "urlopen error",
            "http 408",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "rate_limit",
            "native worker stopped",
            "broken pipe",
        )
    )


def supervise():
    attempts = {}
    stages = {}
    children = {}
    last_counts = None
    while True:
        if (OUT / "supervisor.stop").is_file():
            return
        counts = {
            name: len(list((OUT / folder).glob("[0-9][0-9][0-9].json")))
            for name, (_, folder, _) in STAGES.items()
        }
        for name, (lock, _, script) in STAGES.items():
            count = counts[name]
            stages[name] = "complete" if count == 500 else "running"
            if count == 500 or running(lock):
                continue
            child = children.get(name)
            if child and child.poll() is None:
                stages[name] = "starting"
                continue
            key = (name, count)
            if not restartable(log_tail(name)) or attempts.get(key, 0) >= 3:
                stages[name] = "needs attention"
                continue
            attempts[key] = attempts.get(key, 0) + 1
            print(
                f"Recovering {name} at {count}/500, attempt {attempts[key]}", flush=True
            )
            with (OUT / f"{name}.log").open("a") as log:
                children[name] = subprocess.Popen(
                    [sys.executable, "-u", str(script)],
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            stages[name] = "starting"
        put(
            OUT / "supervisor-status.json",
            {
                "checked": time.time(),
                "counts": counts,
                "stages": stages,
                "restarts": sum(attempts.values()),
            },
        )
        if counts != last_counts:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), counts, stages, flush=True)
            last_counts = counts
        if all(v == 500 for v in counts.values()):
            subprocess.run(
                [sys.executable, ROOT / "scripts/longmemeval-everos-report.py"],
                cwd=ROOT,
                check=True,
            )
            return
        time.sleep(12)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        supervise()
