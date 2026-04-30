"""Stub synthesizer for fetched-photo entries.

Registers the `fetched_photo` content_kind so that the synthesize()
dispatcher raises a clear, actionable error instead of 'unknown content_kind'
when a fetched entry is accidentally routed through the synthesis path.
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
