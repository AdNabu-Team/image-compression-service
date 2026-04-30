"""Build a static HTML dashboard from the git history of reports/baseline.core.json.

Reads every prior version of the baseline via ``git log --follow``, extracts
per-format trend data, and renders ``index.html`` + ``data/history.json`` into
``dashboard/dist/``.  Idempotent — safe to run repeatedly; output dir is wiped
each run.

CLI::

    python -m bench.dashboard.build [--out-dir dashboard/dist] [--limit 100]
    python -m bench.dashboard.build --repo /path/to/repo --out-dir /tmp/dash
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASELINE_PATH = "reports/baseline.core.json"
_SCHEMA_VERSION = 2

# All formats the corpus currently covers (used as column order in history).
KNOWN_FORMATS = [
    "jpeg",
    "png",
    "webp",
    "avif",
    "heic",
    "jxl",
    "gif",
    "apng",
    "bmp",
    "tiff",
    "svg",
    "svgz",
]


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def list_commits(repo_root: Path, file_path: str, limit: int) -> list[dict[str, str]]:
    """Return commits that touched *file_path*, oldest-first.

    Each entry: ``{sha, unix_ts, subject}``.
    """
    result = _git(
        "log",
        "--follow",
        "--format=%H %ct %s",
        "--",
        file_path,
        cwd=repo_root,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    commits: list[dict[str, str]] = []
    for line in result.stdout.strip().splitlines():
        parts = line.split(" ", 2)
        if len(parts) < 2:
            continue
        sha = parts[0]
        unix_ts = parts[1]
        subject = parts[2] if len(parts) > 2 else ""
        commits.append({"sha": sha, "unix_ts": unix_ts, "subject": subject})

    # git log returns newest-first; we want oldest-first for x-axis flow.
    commits.reverse()

    # Apply limit from the newest end (keep the last N after reversal).
    if limit > 0 and len(commits) > limit:
        commits = commits[-limit:]

    return commits


def show_file(repo_root: Path, sha: str, file_path: str) -> str | None:
    """Return the content of *file_path* at *sha*, or None if missing."""
    result = _git("show", f"{sha}:{file_path}", cwd=repo_root, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


# ---------------------------------------------------------------------------
# Run JSON parsing
# ---------------------------------------------------------------------------


def _parse_run(raw: str) -> dict[str, Any] | None:
    """Parse a run JSON string. Returns None on schema mismatch or parse error."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if data.get("schema_version") != _SCHEMA_VERSION:
        return None
    return data


def _aggregate_by_format(stats: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group *stats* entries by format and compute per-format medians.

    Returns a dict keyed by format name::

        {
            "jpeg": {"p50_ms": 23.4, "p95_ms": 45.1, "peak_rss_kb": 70000, "n": 21},
            ...
        }
    """
    by_fmt: dict[str, list[dict[str, Any]]] = {}
    for s in stats:
        fmt = s.get("format", "unknown")
        by_fmt.setdefault(fmt, []).append(s)

    result: dict[str, dict[str, Any]] = {}
    for fmt, entries in by_fmt.items():
        p50_list = [e["p50_ms"] for e in entries if "p50_ms" in e]
        p95_list = [e["p95_ms"] for e in entries if "p95_ms" in e]
        # Peak RSS = max of (parent + children) across cases within this format.
        # We use the p95 values from each stat row to stay consistent with
        # what CaseStats records.
        rss_list = [
            e.get("parent_peak_rss_p95_kb", 0) + e.get("children_peak_rss_p95_kb", 0)
            for e in entries
            if "parent_peak_rss_p95_kb" in e
        ]
        result[fmt] = {
            "p50_ms": round(median(p50_list), 3) if p50_list else 0.0,
            "p95_ms": round(median(p95_list), 3) if p95_list else 0.0,
            "peak_rss_kb": int(median(rss_list)) if rss_list else 0,
            "n": len(entries),
        }
    return result


def extract_run_record(
    sha: str,
    unix_ts: str,
    subject: str,
    run_data: dict[str, Any],
) -> dict[str, Any]:
    """Build the history.json run record from a parsed run JSON."""
    stats: list[dict[str, Any]] = run_data.get("stats", [])
    iterations: list[dict[str, Any]] = run_data.get("iterations", [])

    n_errors = sum(1 for it in iterations if it.get("error") is not None)
    n_cases = len(stats)

    ts_int = int(unix_ts)
    iso_date = dt.datetime.fromtimestamp(ts_int, tz=dt.timezone.utc).strftime("%Y-%m-%d")

    # Short SHA: 7 chars is conventional.
    short_sha = sha[:7]

    return {
        "sha": sha,
        "short_sha": short_sha,
        "timestamp_unix": ts_int,
        "iso_date": iso_date,
        "subject": subject,
        "n_cases": n_cases,
        "n_errors": n_errors,
        "by_format": _aggregate_by_format(stats),
    }


# ---------------------------------------------------------------------------
# History builder
# ---------------------------------------------------------------------------


def build_history(
    repo_root: Path,
    limit: int = 100,
) -> dict[str, Any]:
    """Walk git log and collect one record per historical baseline commit.

    Returns the full ``history.json`` payload (not yet serialized).
    """
    commits = list_commits(repo_root, BASELINE_PATH, limit)
    runs: list[dict[str, Any]] = []

    for commit in commits:
        sha = commit["sha"]
        raw = show_file(repo_root, sha, BASELINE_PATH)
        if raw is None:
            # File didn't exist at this commit — shouldn't happen with
            # ``--follow`` but guard anyway.
            continue
        run_data = _parse_run(raw)
        if run_data is None:
            # Pre-schema-v2 or malformed — skip gracefully.
            continue
        record = extract_run_record(
            sha=sha,
            unix_ts=commit["unix_ts"],
            subject=commit["subject"],
            run_data=run_data,
        )
        runs.append(record)

    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    latest_sha = runs[-1]["short_sha"] if runs else "none"

    return {
        "generated_at": now_iso,
        "manifest": "core",
        "latest_sha": latest_sha,
        "runs": runs,  # oldest-first; guaranteed by list_commits()
    }


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------


def _template_dir() -> Path:
    return Path(__file__).parent / "template"


def render_output(history: dict[str, Any], out_dir: Path) -> None:
    """Write ``index.html`` and ``data/history.json`` to *out_dir*.

    Wipes *out_dir* first so repeated runs stay idempotent.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "data").mkdir()

    # Copy the HTML template verbatim (JS reads history.json at runtime).
    src_html = _template_dir() / "index.html"
    shutil.copy(src_html, out_dir / "index.html")

    # Write the data file.
    json_path = out_dir / "data" / "history.json"
    json_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


def _no_baseline_page(out_dir: Path) -> None:
    """Write a minimal placeholder when there is no baseline history yet."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / "data").mkdir()

    placeholder = {
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": "core",
        "latest_sha": "none",
        "runs": [],
    }
    (out_dir / "data" / "history.json").write_text(
        json.dumps(placeholder, indent=2) + "\n", encoding="utf-8"
    )

    # Still copy the template so a valid page is served.
    src_html = _template_dir() / "index.html"
    if src_html.exists():
        shutil.copy(src_html, out_dir / "index.html")
    else:
        (out_dir / "index.html").write_text(
            "<!doctype html><html><body>"
            "<h1>Pare bench dashboard</h1>"
            "<p>No baseline pinned yet. Run the bench workflow on main first.</p>"
            "</body></html>\n",
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _find_repo_root(start: Path) -> Path:
    """Walk up from *start* until we find a ``.git`` directory."""
    current = start.resolve()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists():
            return parent
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a static HTML dashboard from bench git history.",
        prog="bench.dashboard.build",
    )
    parser.add_argument(
        "--out-dir",
        default="dashboard/dist",
        help="Output directory (wiped and recreated each run). Default: dashboard/dist",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of historical baseline commits to include. Default: 100",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="Path to the git repo root. Defaults to auto-detection from CWD.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve() if args.repo else _find_repo_root(Path.cwd())
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir

    print(f"[dashboard] repo root : {repo_root}", file=sys.stderr)
    print(f"[dashboard] output dir: {out_dir}", file=sys.stderr)

    history = build_history(repo_root, limit=args.limit)
    n_runs = len(history["runs"])

    if n_runs == 0:
        print(
            "[dashboard] No baseline history found — writing placeholder page.",
            file=sys.stderr,
        )
        _no_baseline_page(out_dir)
        return 0

    render_output(history, out_dir)
    print(f"[dashboard] Wrote {n_runs} run(s) to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
