"""Markdown report writer for PR comments and step summaries.

Tables are grouped by `(format, preset)` so format-specific regressions
stand out at a glance. Numbers are formatted with `~3` significant
digits — enough to rank cases without drowning in noise.
"""

from __future__ import annotations

from typing import Any

from bench.runner.stats import CaseStats, percentile

_SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Maps CaseDiff.label values to human-readable display strings for the
# comparison table.  Keep non-regressions as "~" so the table stays scannable.
_COMPARE_LABEL_DISPLAY: dict[str, str] = {
    "significant": "❌ regression",
    "noise_floor_regression": "⚠ noise-floor",
    "improvement": "✅ improvement",
    "below_threshold": "~",
    "noise_floor_ok": "~",
    "ok": "~",
}


def format_compare_label(label: str) -> str:
    """Return the display string for a CaseDiff label column entry."""
    return _COMPARE_LABEL_DISPLAY.get(label, label)


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

    if run["mode"] == "quality":
        out.append("")
        out.append(_render_quality_summary(run["iterations"], config=run.get("config", {})))

    if run["mode"] == "load":
        out.append("")
        out.append(_render_load_summary(run))

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

    mode_label = run["mode"]
    if cfg.get("isolate"):
        mode_label = f"{mode_label} (isolated, fresh subprocess per iteration)"

    lines = [
        f"- **timestamp**: {run['timestamp']}",
        f"- **git**: {git_str}",
        f"- **host**: {run['host']['platform']} ({run['host']['cpu_count']} CPUs)",
        f"- **manifest**: {run['manifest']['name']} (`{run['manifest']['sha256'][:12]}`)",
        f"- **mode**: {mode_label}",
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


def _render_quality_summary(
    iterations: list[dict[str, Any]], *, config: dict[str, Any] | None = None
) -> str:
    """Render a quality-summary section for quality-mode runs.

    Aggregates ssim, psnr_db, ssimulacra2, butteraugli scores across all
    successful lossy cases, then breaks down per-format medians.
    Only called when ``run['mode'] == 'quality'``.
    """
    if config is None:
        config = {}
    # Collect per-case quality metrics from successful (non-failure) rows.
    metric_names = ("ssim", "psnr_db", "ssimulacra2", "butteraugli_max", "butteraugli_3norm")
    all_metrics: dict[str, list[float]] = {m: [] for m in metric_names}
    by_fmt: dict[str, dict[str, list[float]]] = {}

    ssim_total = 0
    ssim_non_null = 0
    ss2_total = 0
    ss2_non_null = 0

    for it in iterations:
        if _is_failure(it):
            continue
        q = it.get("quality")
        if not isinstance(q, dict):
            continue

        fmt = it.get("format", "?")
        by_fmt.setdefault(fmt, {m: [] for m in metric_names})

        for mname in metric_names:
            val = q.get(mname)
            if val is not None:
                try:
                    fval = float(val)
                except (TypeError, ValueError):
                    continue
                all_metrics[mname].append(fval)
                by_fmt[fmt][mname].append(fval)

        # Track null counts for binary-missing detection
        ssim_total += 1
        if q.get("ssim") is not None:
            ssim_non_null += 1
        ss2_total += 1
        if q.get("ssimulacra2") is not None:
            ss2_non_null += 1

    n_scored = ssim_total
    fast_mode = bool(config.get("quality_fast"))

    if n_scored == 0:
        if fast_mode:
            return "## Quality summary (fast mode — pure-numpy SSIM/PSNR only)\n\n_No quality data available._"
        return "## Quality summary\n\n_No quality data available._"

    lines: list[str] = []
    if fast_mode:
        lines.append("## Quality summary (fast mode — pure-numpy SSIM/PSNR only)")
    else:
        lines.append("## Quality summary")
    lines.append("")
    lines.append(f"_{n_scored} lossy case(s) scored._")
    lines.append("")

    # Missing-binary warnings — only fire when subprocess metrics were
    # actually requested (i.e. not suppressed by fast mode).
    if not fast_mode:
        warnings: list[str] = []
        if ss2_non_null == 0 and ss2_total > 0:
            warnings.append("_`ssimulacra2` binary not found; install libjxl tools to enable_")
        if len(all_metrics["butteraugli_max"]) == 0 and n_scored > 0:
            warnings.append("_`butteraugli_main` binary not found; install libjxl tools to enable_")
        for w in warnings:
            lines.append(f"> {w}")
        if warnings:
            lines.append("")

    # Overall aggregates table
    lines.append("### Overall metrics (all formats)")
    lines.append("")
    lines.append("| metric | higher better | n | median | p95 |")
    lines.append("|---|---|---|---|---|")

    def _fmt_metric(vals: list[float], name: str) -> tuple[str, str]:
        if not vals:
            return "-", "-"
        return f"{percentile(vals, 50):.4f}", f"{percentile(vals, 95):.4f}"

    metric_display = [
        ("ssim", "yes", all_metrics["ssim"]),
        ("psnr_db", "yes", all_metrics["psnr_db"]),
        ("ssimulacra2", "yes", all_metrics["ssimulacra2"]),
        ("butteraugli_max", "no", all_metrics["butteraugli_max"]),
        ("butteraugli_3norm", "no", all_metrics["butteraugli_3norm"]),
    ]
    for mname, higher, vals in metric_display:
        med, p95 = _fmt_metric(vals, mname)
        n = len(vals)
        lines.append(f"| `{mname}` | {higher} | {n} | {med} | {p95} |")

    lines.append("")

    # Per-format ssimulacra2 breakdown (where available and ≥1 case)
    if all_metrics["ssimulacra2"]:
        lines.append("### Per-format median `ssimulacra2` (higher = better quality)")
        lines.append("")
        lines.append("| format | n | median | p95 |")
        lines.append("|---|---|---|---|")
        for fmt in sorted(by_fmt.keys()):
            vals = by_fmt[fmt].get("ssimulacra2", [])
            if not vals:
                continue
            med = percentile(vals, 50)
            p95v = percentile(vals, 95)
            lines.append(f"| {fmt} | {len(vals)} | {med:.2f} | {p95v:.2f} |")
        lines.append("")

    # Per-format ssim breakdown
    if all_metrics["ssim"]:
        lines.append("### Per-format median `ssim`")
        lines.append("")
        lines.append("| format | n | median | p95 |")
        lines.append("|---|---|---|---|")
        for fmt in sorted(by_fmt.keys()):
            vals = by_fmt[fmt].get("ssim", [])
            if not vals:
                continue
            med = percentile(vals, 50)
            p95v = percentile(vals, 95)
            lines.append(f"| {fmt} | {len(vals)} | {med:.4f} | {p95v:.4f} |")

    return "\n".join(lines)


def _render_load_summary(run: dict[str, Any]) -> str:
    """Render an aggregate load-summary section for load-mode runs.

    Summarises across all cases:
    - Total requests, 503s, errors and overall ok_rate
    - Median throughput and p95 request latency
    - Gate configuration that produced the numbers
    - Per-format breakdown (median throughput and 503 rate)

    Only called when ``run['mode'] == 'load'``.
    """
    iterations = run.get("iterations", [])
    cfg = run.get("config", {})

    # Collect per-case load blocks from successful (non-failure) rows.
    total_requests = 0
    total_success = 0
    total_503 = 0
    total_error = 0
    throughputs: list[float] = []
    latency_p95s: list[float] = []
    by_fmt: dict[str, dict[str, list[float]]] = {}

    for it in iterations:
        if _is_failure(it):
            continue
        lb = it.get("load")
        if not isinstance(lb, dict):
            continue
        n_con = lb.get("n_concurrent", 0)
        n_ok = lb.get("n_success", 0)
        n_503 = lb.get("n_503", 0)
        n_err = lb.get("n_error", 0)
        total_requests += n_con
        total_success += n_ok
        total_503 += n_503
        total_error += n_err

        tp = lb.get("throughput_per_sec", 0.0)
        throughputs.append(tp)

        lat_block = lb.get("request_latency_ms", {})
        p95 = lat_block.get("p95", 0.0)
        latency_p95s.append(p95)

        fmt = it.get("format", "?")
        by_fmt.setdefault(fmt, {"throughputs": [], "ok_rates": []})
        by_fmt[fmt]["throughputs"].append(tp)
        ok_rate = lb.get("ok_rate", 0.0)
        by_fmt[fmt]["ok_rates"].append(ok_rate)

    if total_requests == 0:
        return "## Load summary\n\n_No load data available._"

    overall_ok_rate = total_success / total_requests if total_requests > 0 else 0.0
    med_throughput = percentile(throughputs, 50) if throughputs else 0.0
    med_lat_p95 = percentile(latency_p95s, 50) if latency_p95s else 0.0

    sem_size = cfg.get("semaphore_size", "?")
    queue_depth = cfg.get("queue_depth", "?")
    n_concurrent = cfg.get("n_concurrent", "?")

    lines: list[str] = []
    lines.append("## Load summary")
    lines.append("")
    lines.append(
        f"_Gate config: `semaphore_size={sem_size}`, `queue_depth={queue_depth}`, "
        f"`n_concurrent={n_concurrent}` per case._"
    )
    lines.append("")

    # Aggregate totals table
    lines.append("### Aggregate totals")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| total requests | {total_requests} |")
    lines.append(f"| successful | {total_success} ({overall_ok_rate:.1%}) |")
    lines.append(f"| 503 (backpressure) | {total_503} |")
    lines.append(f"| other errors | {total_error} |")
    lines.append(f"| median throughput | {med_throughput:.1f} req/s |")
    lines.append(f"| median p95 latency | {_fmt_ms(med_lat_p95)} |")
    lines.append("")

    # Per-format breakdown
    if by_fmt:
        lines.append("### Per-format breakdown")
        lines.append("")
        lines.append("| format | cases | median throughput | median ok_rate |")
        lines.append("|---|---|---|---|")
        for fmt in sorted(by_fmt.keys()):
            tps = by_fmt[fmt]["throughputs"]
            ors = by_fmt[fmt]["ok_rates"]
            med_tp = percentile(tps, 50) if tps else 0.0
            med_or = percentile(ors, 50) if ors else 0.0
            lines.append(f"| {fmt} | {len(tps)} | {med_tp:.1f} req/s | {med_or:.1%} |")

    return "\n".join(lines)


def _stats_from_run(run: dict[str, Any]) -> list[CaseStats]:
    """Reconstruct CaseStats from the JSON payload's `stats` array."""
    rebuilt = []
    for s in run.get("stats", []):
        rebuilt.append(CaseStats(**s))
    return rebuilt
