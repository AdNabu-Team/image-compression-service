# bench/

Pare benchmarking + corpus toolkit. Replaces the legacy `benchmarks/` and `scripts/{download,convert}_corpus*.py`.

## Why this exists separately from `benchmarks/`

The legacy `benchmarks/` measures `time.process_time()` (parent only) and `tracemalloc` (Python heap only). MozJPEG, pngquant, oxipng, cjxl etc. run as subprocesses and account for 80–95 % of CPU work, so the legacy CPU and memory numbers under-report by an order of magnitude. `bench/` uses `RUSAGE_CHILDREN` deltas and `ru_maxrss` of children to capture honest totals.

## Two-tier corpus

The corpus has two named manifests that can be benchmarked independently or together:

**`core` — synthesized** (`bench/corpus/manifests/core.json`, manifest_version=1):
- Images generated deterministically from `(kind, seed, dims)` via `bench/corpus/synthesis/`.
- No network required; reproducible anywhere from source code alone.
- Pixel-level SHA-256 is pinned in the manifest (`expected_pixel_sha256`).

**`full` — fetched + hash-pinned** (`bench/corpus/manifests/full.json`, manifest_version=2):
- Real-world images downloaded from declared URLs (currently: Kodak Lossless True Color Image Suite, 8 of 24 images).
- Each entry has a `source` field with `url`, `sha256`, `license`, and `attribution`.
- Fetched bytes are cached under `bench/corpus/cache/<sha256[:2]>/<sha256>/<basename>` — not committed; add to `.gitignore`.
- Pixel-level SHA-256 is still pinned per entry — the pixel hash catches CDN re-encodes.
- Use `python -m bench.corpus fetch --manifest full` to pre-warm the cache before building.

### v1 → v2 schema change

`MANIFEST_VERSION` was bumped from `1` to `2` in `bench/corpus/manifest.py`. `Manifest.from_json()` accepts both `{1, 2}`. v1 entries have no `source` field (`source=None` after loading). No structural changes to existing synthesized entries are required.

A new `SourceSpec` dataclass was added with fields: `url`, `sha256`, `license`, `attribution`, `notes` (optional). A new `fetched_photo` content_kind is registered as a stub that raises `RuntimeError` if synthesis is accidentally invoked — the builder routes fetched entries through `fetch()` instead.

## Subpackages

- **`bench.corpus`** — manifest-driven corpus builder. Synthesizers produce deterministic pixel data from seeds; fetchers download hash-pinned real-world images. Manifests pin pixel-level SHA-256 (decoded `Image.tobytes()`), not encoded SHA — encoded bytes drift across libjpeg-turbo SIMD paths.
- **`bench.corpus.fetchers`** — HTTP fetcher (`bench/corpus/fetchers/http.py`). Downloads to a content-addressed local cache (`bench/corpus/cache/`); verifies SHA-256 before returning the path. Exceptions: `FetchError`, `FetchIntegrityError`, `FetchHTTPError`, `FetchTooLargeError`. Default cache root overridable via `--cache PATH`.
- **`bench.runner`** — subprocess-aware benchmark runner. Modes: `quick` (1 iter, smoke), `timing` (5 iter + warmup, p50/p95/p99 + MAD, `--isolate`), `memory` (1 iter, peak RSS headline). `load` mode deferred to v1.

## Determinism contract

- Pixel-level SHA-256 of raw `Image.tobytes()` is canonical.
- Encoded SHA may be recorded as `encoded_sha256.<platform>` for diagnostics, but is never blocking.
- `random.Random(seed)` instances only — never mutate the global PRNG.
- Fonts vendored at `bench/corpus/fonts/` (Pillow's default font is build-dependent).
- Fetched entries: pixel hash is computed from the decoded source image (same `pixel_sha256()` function). The source URL's SHA-256 guards against corrupt downloads; the pixel SHA-256 guards against CDN re-encoding.

## Common commands

```bash
# Build the canonical synthetic corpus
python -m bench.corpus build --manifest core

# Verify the on-disk corpus matches the manifest pixel-hashes
python -m bench.corpus verify --manifest core

# Pre-warm the fetcher cache (downloads all URLs, verifies hashes)
python -m bench.corpus fetch --manifest full

# Build the fetched corpus (uses cached files; fetches on cache miss)
python -m bench.corpus build --manifest full --seal   # first time: seal pixel hashes
python -m bench.corpus build --manifest full           # subsequent builds

# Verify the fetched corpus
python -m bench.corpus verify --manifest full

# Override fetcher cache location
python -m bench.corpus fetch --manifest full --cache /path/to/cache
python -m bench.corpus build --manifest full --cache /path/to/cache

# Quick smoke (1 iter, all formats, ~1 min)
python -m bench.run --mode quick

# Honest latency (5 iters, isolated, p50/p95/p99)
python -m bench.run --mode timing --manifest core --out reports/timing.json

# Peak RSS truth (1 iter, isolated)
python -m bench.run --mode memory --out reports/memory.json

# Diff two runs with Welch's t-test
python -m bench.compare reports/baseline.json reports/head.json --threshold-pct 10
```

## Code style

Same as repo root: Black 100 cols, Ruff E/F/W/I, Python 3.12, pytest-asyncio.
