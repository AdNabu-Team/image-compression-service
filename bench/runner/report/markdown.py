"""Markdown report writer for PR comments and step summaries.

Tables are grouped by `(format, preset)` so format-specific regressions
stand out at a glance. Numbers are formatted with `~3` significant
digits — enough to rank cases without drowning in noise.
"""

from __future__ import annotations

from typing import Any

from bench.runner.stats import CaseStats, percentile

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


def _sparkline(samples: list, max_buckets: int = 24) -> str:
    """Compress an RSS sample series into a unicode sparkline.

    Each sample is `[offset_ms, rss_kb]`. We drop the time axis and bucket
    by max-rss so a single ASCII glyph reflects the worst case in its
    window — that matches what readers want to see at a glance.
    """
    if not samples:
        return ""
    values = [s[1] for s in samples if isinstance(s, (list, tuple)) and len(s) >= 2]
    if not values:
        return ""
    if len(values) > max_buckets:
        # Fold to max_buckets via max-pooling.
        bucket_size = len(values) / max_buckets
        bucketed: list[float] = []
        for i in range(max_buckets):
            start = int(i * bucket_size)
            end = max(start + 1, int((i + 1) * bucket_size))
            bucketed.append(max(values[start:end]))
        values = bucketed
    lo, hi = min(values), max(values)
    if hi == lo:
        return _SPARK_CHARS[len(_SPARK_CHARS) // 2] * len(values)
    return "".join(
        _SPARK_CHARS[
            min(len(_SPARK_CHARS) - 1, int((v - lo) / (hi - lo) * (len(_SPARK_CHARS) - 1)))
        ]
        for v in values
    )


def _fmt_ms(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.2f}s"
    if v >= 100:
        return f"{v:.0f}ms"
    if v >= 10:
        return f"{v:.1f}ms"
    return f"{v:.2f}ms"


def _fmt_kb(v: int | None) -> str:
    if v is None:
        return "-"
    if v >= 1024 * 1024:
        return f"{v / (1024 * 1024):.1f}GB"
    if v >= 1024:
        return f"{v / 1024:.1f}MB"
    return f"{v}KB"


def _is_failure(it: dict[str, Any]) -> bool:
    """Return True if this iteration represents a run-time failure.

    A failure row has a top-level ``error`` field — either a string (legacy
    quick/timing/memory shape) or a dict with ``phase`` (accuracy shape).
    Successful accuracy rows carry their prediction-error metrics under
    ``accuracy``, never ``error``.
    """
    err = it.get("error")
    if err is None:
        return False
    return isinstance(err, (str, dict))


def _failure_summary(it: dict[str, Any]) -> str:
    """Return a short human-readable failure description."""
    err = it.get("error")
    if isinstance(err, str):
        return err
    if isinstance(err, dict):
        phase = err.get("phase", "?")
        msg = err.get("message", "unknown error")
        return f"[{phase}] {msg}"
    return repr(err)


def render_run(run: dict[str, Any]) -> str:
    """Render a full run JSON payload as Markdown."""
    out: list[str] = []
    out.append(f"# Pare bench — `{run['mode']}` mode")
    out.append("")
    out.append(_render_metadata(run))
    out.append("")

    failures = [it for it in run["iterations"] if _is_failure(it)]
    if failures:
        out.append(f"## Errors ({len(failures)})")
        out.append("")
        for e in failures[:10]:
            out.append(f"- `{e['case_id']}`: {_failure_summary(e)}")
        if len(failures) > 10:
            out.append(f"- … {len(failures) - 10} more")
        out.append("")

    stats = _stats_from_run(run)
    if not stats:
        out.append("_No successful iterations to report._")
        return "\n".join(out)

    out.append("## Per-case results")
    out.append("")
    out.append(_render_stats_table(stats))

    if run["mode"] == "memory":
        out.append("")
        out.append("## Memory headline (peak RSS — capacity planning)")
        out.append("")
        out.append(_render_memory_table(stats))

    if run["mode"] == "accuracy":
        out.append("")
        out.append(_render_accuracy_summary(run["iterations"]))

    return "\n".join(out)


def _render_metadata(run: dict[str, Any]) -> str:
    git = run.get("git", {})
    git_str = f"{git.get('branch', '?')} @ {git.get('commit', '?')[:8]}" + (
        " (dirty)" if git.get("dirty") else ""
    )
    annotations = run.get("annotations") or {}
    ann_lines = "\n".join(f"- **{k}**: {v}" for k, v in annotations.items())

    cfg = run.get("config", {})
    cfg_pairs = ", ".join(f"{k}={v}" for k, v in cfg.items())

    lines = [
        f"- **timestamp**: {run['timestamp']}",
        f"- **git**: {git_str}",
        f"- **host**: {run['host']['platform']} ({run['host']['cpu_count']} CPUs)",
        f"- **manifest**: {run['manifest']['name']} (`{run['manifest']['sha256'][:12]}`)",
        f"- **config**: {cfg_pairs}",
    ]
    if ann_lines:
        lines.append(ann_lines)
    return "\n".join(lines)


def _render_stats_table(stats: list[CaseStats]) -> str:
    lines = [
        "| case_id | iter | p50 | p95 | median±MAD | child CPU p50 | parallel | RSS p95 | red% | method |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in sorted(stats, key=lambda x: (x.format, x.preset, x.case_id)):
        lines.append(
            "| `{cid}` | {it} | {p50} | {p95} | {med}±{mad} | {ccpu} | {par:.2f}× | {rss} | {red:.1f}% | {meth} |".format(
                cid=s.case_id,
                it=s.iterations,
                p50=_fmt_ms(s.p50_ms),
                p95=_fmt_ms(s.p95_ms),
                med=_fmt_ms(s.median_ms),
                mad=_fmt_ms(s.mad_ms),
                ccpu=_fmt_ms(s.children_cpu_p50_ms),
                par=s.parallelism_p50,
                rss=_fmt_kb(s.children_peak_rss_p95_kb),
                red=s.reduction_pct,
                meth=(s.method or "-")[:24],
            )
        )
    return "\n".join(lines)


def _render_memory_table(stats: list[CaseStats]) -> str:
    lines = [
        "| case_id | parent peak | children peak | py heap peak | curve peak (samples) | spark |",
        "|---|---|---|---|---|---|",
    ]
    for s in sorted(
        stats, key=lambda x: -max(x.parent_peak_rss_p95_kb, x.children_peak_rss_p95_kb)
    ):
        samples = s.rss_samples or []
        if samples:
            sample_peak_kb = max(int(pt[1]) for pt in samples)
            curve_summary = f"{_fmt_kb(sample_peak_kb)} ({len(samples)})"
            spark = _sparkline(samples)
        else:
            curve_summary = "-"
            spark = "-"
        lines.append(
            f"| `{s.case_id}` | {_fmt_kb(s.parent_peak_rss_p95_kb)} "
            f"| {_fmt_kb(s.children_peak_rss_p95_kb)} "
            f"| {_fmt_kb(s.py_peak_alloc_p95_kb)} "
            f"| {curve_summary} | `{spark}` |"
        )
    return "\n".join(lines)


def _render_accuracy_summary(iterations: list[dict[str, Any]]) -> str:
    """Render an accuracy-summary section for accuracy-mode runs.

    Aggregates reduction_abs_error_pct_abs and size_rel_error_pct across
    all successful cases, then breaks down median error per output format.
    Only called when ``run['mode'] == 'accuracy'``.
    """
    # Collect per-case error metrics from successful (non-failure) rows.
    red_abs_errs: list[float] = []
    size_rel_errs: list[float] = []
    by_fmt: dict[str, list[float]] = {}
    no_op_correct = 0
    no_op_total = 0

    for it in iterations:
        if _is_failure(it):
            continue
        acc = it.get("accuracy")
        if not isinstance(acc, dict):
            continue
        red_abs = acc.get("reduction_abs_error_pct_abs")
        size_rel = acc.get("size_rel_error_pct")
        if red_abs is None or size_rel is None:
            continue
        red_abs_errs.append(float(red_abs))
        size_rel_errs.append(abs(float(size_rel)))
        fmt = it.get("format", "?")
        by_fmt.setdefault(fmt, []).append(float(red_abs))

        # Check if already_optimized prediction matched actual < 1% reduction
        est = it.get("estimate", {})
        opt = it.get("optimize", {})
        predicted_already_opt = est.get("already_optimized", False)
        actual_reduction = opt.get("actual_reduction_pct", 0.0)
        actual_no_op = actual_reduction < 1.0
        if predicted_already_opt:
            no_op_total += 1
            if actual_no_op:
                no_op_correct += 1

    if not red_abs_errs:
        return "## Accuracy summary\n\n_No accuracy data available._"

    lines: list[str] = []
    lines.append("## Accuracy summary")
    lines.append("")
    lines.append(
        f"_{len(red_abs_errs)} successful case(s) measured. "
        f"Positive size_rel_error = estimator overestimates (predicted size > actual)._"
    )
    lines.append("")

    # Overall aggregates
    med_red = percentile(red_abs_errs, 50)
    p95_red = percentile(red_abs_errs, 95)
    med_size = percentile(size_rel_errs, 50)
    p95_size = percentile(size_rel_errs, 95)
    lines.append("### Overall error")
    lines.append("")
    lines.append("| metric | median | p95 |")
    lines.append("|---|---|---|")
    lines.append(f"| `reduction_abs_error_pct_abs` | {med_red:.2f}% | {p95_red:.2f}% |")
    lines.append(f"| `abs(size_rel_error_pct)` | {med_size:.2f}% | {p95_size:.2f}% |")
    lines.append("")

    # No-op identification accuracy
    if no_op_total > 0:
        lines.append(
            f"**No-op identification**: {no_op_correct}/{no_op_total} cases where "
            f"`already_optimized=true` correctly matched `actual_reduction_pct < 1.0`"
        )
        lines.append("")

    # Per-format breakdown
    lines.append("### Per-format median `reduction_abs_error_pct_abs`")
    lines.append("")
    lines.append("| format | n | median abs error |")
    lines.append("|---|---|---|")
    for fmt in sorted(by_fmt.keys()):
        vals = by_fmt[fmt]
        med = percentile(vals, 50)
        lines.append(f"| {fmt} | {len(vals)} | {med:.2f}% |")

    return "\n".join(lines)


def _stats_from_run(run: dict[str, Any]) -> list[CaseStats]:
    """Reconstruct CaseStats from the JSON payload's `stats` array."""
    rebuilt = []
    for s in run.get("stats", []):
        rebuilt.append(CaseStats(**s))
    return rebuilt
