"""Benchmark run modes.

quick   — 1 iteration per case, all formats, ~1 minute. PR sanity check.
timing  — 5 iterations + 1 warmup, --isolate, p50/p95/p99 + MAD.
memory  — 1 iteration + --isolate, max(parent, children) peak RSS as headline.
load    — N concurrent requests per case through a fresh CompressionGate;
          measures throughput, 503 rate, latency tail under contention.
"""
