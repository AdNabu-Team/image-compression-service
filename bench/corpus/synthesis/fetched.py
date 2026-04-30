"""Stub synthesizers for fetched entries.

Registers the `fetched_photo` and `fetched_vector` content_kinds so that the
synthesize() dispatcher raises a clear, actionable error instead of 'unknown
content_kind' when a fetched entry is accidentally routed through the synthesis
path.

For `fetched_vector` (SVG/SVGZ): the builder fetches the source bytes directly
and passes them to the encoder as raw bytes — no Image.open() involved.
"""

from __future__ import annotations

from bench.corpus.synthesis._common import register_kind


@register_kind("fetched_photo")
def _fetched_photo_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_photo' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec instead. If you reached this "
        "via a manual synthesize() call, route through the build() pipeline."
    )


@register_kind("fetched_vector")
def _fetched_vector_stub(*, seed: int, width: int, height: int, **params) -> None:  # type: ignore[return]
    raise RuntimeError(
        "content_kind 'fetched_vector' is not synthesized — the builder must "
        "fetch this entry's bytes via SourceSpec and write them as-is (vector "
        "pass-through). If you reached this via a manual synthesize() call, "
        "route through the build() pipeline."
    )
