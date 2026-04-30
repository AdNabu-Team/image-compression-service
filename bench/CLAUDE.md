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

A new `SourceSpec` dataclass was added with fields: `url`, `sha256`, `license`, `attribution`, `notes` (optional). Two stubs are registered: `fetched_photo` (raster) and `fetched_vector` (vector) — both raise `RuntimeError` if synthesis is accidentally invoked; the builder routes these through `fetch()` instead.

A new `expected_byte_sha256` field (`dict[str, str] | None`) was added to `ManifestEntry` for vector entries. Raster entries leave this `None` and use `expected_pixel_sha256` as before. Vector entries leave `expected_pixel_sha256` as `None` and store `{"source": "<sha256>"}` in `expected_byte_sha256`.

## Subpackages

- **`bench.corpus`** — manifest-driven corpus builder. Synthesizers produce deterministic pixel data from seeds; fetchers download hash-pinned real-world images. Raster manifests pin pixel-level SHA-256 (decoded `Image.tobytes()`); vector manifests (SVG/SVGZ) pin byte-level SHA-256 of the source file. Encoded raster bytes drift across libjpeg-turbo SIMD paths and are never the canonical hash; vector encoded bytes are deterministic (mtime=0 gzip) and can be hashed but the source SHA is the contract.
- **`bench.corpus.fetchers`** — HTTP fetcher (`bench/corpus/fetchers/http.py`). Downloads to a content-addressed local cache (`bench/corpus/cache/`); verifies SHA-256 before returning the path. Exceptions: `FetchError`, `FetchIntegrityError`, `FetchHTTPError`, `FetchTooLargeError`. Default cache root overridable via `--cache PATH`.
- **`bench.runner`** — subprocess-aware benchmark runner. Modes: `quick` (1 iter, smoke), `timing` (5 iter + warmup, p50/p95/p99 + MAD, `--isolate`), `memory` (1 iter, peak RSS headline). `load` mode deferred to v1.

## Determinism contract

- **Raster entries**: Pixel-level SHA-256 of raw `Image.tobytes()` is canonical (field `expected_pixel_sha256`). Encoded SHA may be recorded as `encoded_sha256.<platform>` for diagnostics, but is never blocking.
- **Vector entries** (SVG/SVGZ): Byte-level SHA-256 of the raw source bytes is canonical (field `expected_byte_sha256["source"]`). SVG sources are XML; no pixel data exists. Encoded bytes are deterministic across platforms (no SIMD variance), so a flat `{format: sha256}` mapping suffices.
- `random.Random(seed)` instances only — never mutate the global PRNG.
- Fonts vendored at `bench/corpus/fonts/` (Pillow's default font is build-dependent).
- Fetched raster entries: pixel hash computed from decoded source image (same `pixel_sha256()` function). Source URL SHA-256 guards against corrupt downloads; pixel SHA-256 guards against CDN re-encoding.
- Fetched vector entries (`fetched_vector` content_kind): source SHA-256 guards against corrupt downloads. The builder writes bytes directly to disk — no `Image.open()` is called.

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

## CI integration

The workflow `.github/workflows/bench-pr.yml` runs automatically on pull requests that touch `optimizers/`, `estimation/`, `utils/`, `bench/`, `schemas.py`, `requirements.txt`, `Dockerfile`, or the workflow file itself.

**Baseline location**: `reports/baseline.core.json` — checked into the repo (unignored via `.gitignore`).

**What CI does**:
1. Builds the Docker image from the PR's source tree (with Buildx GHA layer cache for speed).
2. Inside the container, runs `python -m bench.corpus build --manifest core` then `python -m bench.run --mode quick --manifest core --out reports/_head.json`.
3. Runs `python -m bench.compare reports/baseline.core.json reports/_head.json --threshold-pct 10 --format markdown`.
4. Posts (or updates) a PR comment with the diff table. A hidden HTML signature `<!-- pare-bench-comment -->` ensures repeat pushes update the same comment rather than creating new ones.
5. Fails the workflow if `bench.compare` exits non-zero (regression in at least one case exceeds ±10%).

**Threshold**: ±10% change in median `wall_ms`. Both Welch's t-test (α=0.05) and Cohen's d (≥0.5) must clear before a case is flagged — see `bench/runner/compare.py` for the full logic.

**No-baseline path**: If `reports/baseline.core.json` is missing, the workflow posts a "no baseline; this run is the candidate baseline" comment and does not fail.

**How to refresh the baseline** (run locally, then commit the result):

```bash
python -m bench.corpus build --manifest core
python -m bench.run --mode quick --manifest core \
  --annotate "env=local-venv-bootstrap" \
  --out reports/baseline.core.json
git add reports/baseline.core.json
git commit -m "chore(bench): refresh baseline.core.json"
```

Refresh the baseline whenever you intentionally change optimizer behavior, add corpus entries, or when you want to adopt the Docker-built numbers as the new reference (run the CI workflow on a clean branch, pull `reports/_head.json` from the artifact, rename it, and commit).

## Code style

Same as repo root: Black 100 cols, Ruff E/F/W/I, Python 3.12, pytest-asyncio.
