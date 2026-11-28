"""Loopback-only dashboard for an active longmemeval_qwen.py run."""

import argparse
import fcntl
import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import locomo_progress
import longmemeval_everos_progress
from locomo_mps_report import report as gpu_report

PAGE = Path(__file__).with_name("longmemeval-progress.html")


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def locked(path):
    if not path.is_file():
        return False
    with path.open("rb") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def snapshot(out):
    r, q = (
        read(out / name) for name in ("retrieval-progress.json", "qa-progress.json")
    )
    n = r.get("total", 500)
    items = [{} for _ in range(n)]
    for name, field in [
        ("rankings", "rank"),
        ("answers", "answer"),
        ("judgments", "judge"),
    ]:
        for p in (out / name).glob("*.json"):
            index = int(p.stem)
            if not 0 <= index < n:
                continue
            row = read(p)
            item = items[index]
            item["id"] = row["question_id"]
            item[field] = True
            if field == "rank":
                item.update(
                    kind=row["question_type"],
                    candidates=row["candidates"],
                    ranking=row["methods"]["qwen"]["positions"],
                )
            if field == "judge":
                item["correct"] = row["autoeval_label"]["label"]
    for k in r.get("errors", {}):
        items[int(k) - 1]["error"] = True
    last = ""
    log = out / "worker.log"
    if log.is_file():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 1500))
            lines = f.read().decode(errors="replace").splitlines()
        last = next((v for v in reversed(lines) if v.startswith("question ")), "")
    return {
        "time": datetime.now(UTC).isoformat(),
        "total": n,
        "retrieval": r,
        "qa": q,
        "items": items,
        "worker": last,
        "retrieving": locked(out / ".lock"),
        "grading": locked(out / ".qa-lock"),
    }


def v2_snapshot(root):
    setup = read(root / "progress.json")
    domains = []
    for name in ("web", "enterprise"):
        folder = root / "runs" / f"memoryh_{name}_small" / "progress"
        counts = {
            stage: len(list((folder / stage).glob("*.json")))
            for stage in ("indexed", "queries", "answers")
        }
        judgments = [read(path) for path in (folder / "judgments").glob("*.json")]
        domains.append(
            {
                "name": name,
                "total": setup["totals"][name],
                "indexed": counts["indexed"],
                "queries": counts["queries"],
                "answers": counts["answers"],
                "judged": len(judgments),
                "correct": sum(bool(j["correct"]) for j in judgments),
            }
        )
    systems = []
    for system in read(root / "leaderboard.json").get("systems", []):
        result = system.get("results", {}).get("small")
        if result and system["role"] == "submission":
            point = max(result["operatingPoints"], key=lambda p: p["accuracy"])
            systems.append(
                {
                    "name": system["name"],
                    "accuracy": point["accuracy"],
                    "latency": point["latency"],
                    "gain": result["lafsGain"],
                }
            )
    leaders = sorted(systems, key=lambda s: (-s["gain"], -s["accuracy"]))[:3]
    return dict(
        **setup,
        time=datetime.now(UTC).isoformat(),
        domains=domains,
        leaders=leaders,
        memoryh=read(root / "result.json") or {"name": "MemoryH（本地评测）"},
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            v2 = args.out.resolve().parent / "lme-v2"
            locomo = v2.parent / "locomo-compare"
            everos = v2.parent / "lme-s-everos-20261128"
            if path == "/everos" and (everos / "retrieval-status.json").is_file():
                body = PAGE.with_name("longmemeval-everos-progress.html").read_bytes()
                typ = "text/html; charset=utf-8"
            elif (
                path == "/everos-status.json"
                and (everos / "retrieval-status.json").is_file()
            ):
                body = json.dumps(
                    longmemeval_everos_progress.snapshot(everos, locked)
                ).encode()
                typ = "application/json; charset=utf-8"
            elif path in ("/", "/locomo") and (locomo / "reference.json").is_file():
                body = PAGE.with_name("locomo-progress.html").read_bytes()
                typ = "text/html; charset=utf-8"
            elif path == "/locomo-status.json":
                body = json.dumps(
                    locomo_progress.snapshot(locomo), ensure_ascii=False
                ).encode()
                typ = "application/json; charset=utf-8"
            elif path == "/locomo-yuanbao-gpu.json":
                body = json.dumps(gpu_report(locomo)).encode()
                typ = "application/json; charset=utf-8"
            elif path == "/locomo-questions.csv":
                body, typ = (
                    locomo_progress.per_question_csv(locomo),
                    "text/csv; charset=utf-8",
                )
            elif path in ("/", "/v2") and (v2 / "progress.json").is_file():
                body, typ = (
                    PAGE.with_name("longmemeval-v2-progress.html").read_bytes(),
                    "text/html; charset=utf-8",
                )
            elif path == "/progress.css":
                body = (
                    PAGE.read_bytes().split(b"<style>", 1)[1].split(b"</style>", 1)[0]
                )
                typ = "text/css; charset=utf-8"
            elif path == "/v2-status.json" and (v2 / "progress.json").is_file():
                body, typ = (
                    json.dumps(v2_snapshot(v2), ensure_ascii=False).encode(),
                    "application/json",
                )
            elif path in ("/", "/v1"):
                body, typ = PAGE.read_bytes(), "text/html; charset=utf-8"
            elif path == "/status.json":
                body, typ = (
                    json.dumps(snapshot(args.out), ensure_ascii=False).encode(),
                    "application/json; charset=utf-8",
                )
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", typ)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{srv.server_port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
