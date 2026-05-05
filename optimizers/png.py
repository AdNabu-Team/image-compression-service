import asyncio

import oxipng

from optimizers.base import BaseOptimizer
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat, is_apng
from utils.metadata import strip_metadata_selective
from utils.subprocess_runner import run_tool


class PngOptimizer(BaseOptimizer):
    """PNG optimization: pngquant (lossy) + oxipng (lossless).

    Pipeline:
    1. If APNG → oxipng only (pngquant destroys animation frames)
    2. If png_lossy=False → oxipng only (user requested lossless)
    3. Otherwise → pngquant → oxipng on result
    4. pngquant exit code 99 (quality threshold not met) → fallback to oxipng on original

    Quality controls aggressiveness:
    - quality < 50:  64 max colors, floor=1, speed=3, oxipng level=4 (aggressive)
    - quality < 70:  256 max colors, floor=1, speed=4, oxipng level=4 (moderate)
    - quality >= 70: lossless only, oxipng level=2 (gentle)

    APNG path always uses oxipng level=2 regardless of quality preset. Level 4's
    24 filter trials per frame add ~2s on xlarge animations with minimal gain:
    frame chunks share dictionaries, so filter optimization yields less than for
    static PNG where every byte is in a single IDAT stream.
    """

    format = ImageFormat.PNG

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        animated = is_apng(data)
        if animated:
            self.format = ImageFormat.APNG

        # Strip metadata first (preserves iCCP, pHYs; strips tEXt chunks)
        if config.strip_metadata:
            data_clean = strip_metadata_selective(
                data,
                ImageFormat.APNG if animated else ImageFormat.PNG,
            )
        else:
            data_clean = data

        # Quality-dependent oxipng level: higher = slower but better compression
        # Level 6 = 180 filter trials (too slow for API use on large images)
        # Level 4 = 24 trials (good tradeoff for aggressive/moderate presets)
        # Level 3 is NOT used — it misses critical filters for screenshots
        if config.quality < 70:
            oxipng_level = 4
        else:
            oxipng_level = 2

        # APNG or lossless-only: skip pngquant
        if animated or not config.png_lossy:
            # APNG: always use level 2 — level 4's 24 filter trials per frame
            # add ~2s on xlarge animations with little gain (frames share
            # dictionaries, so filter optimization matters less than for static PNG).
            level = 2 if animated else oxipng_level
            optimized = await asyncio.to_thread(self._run_oxipng, data_clean, level)
            return self._build_result(data, optimized, "oxipng")

        # Quality-dependent pngquant settings
        if config.quality < 50:
            max_colors = 64
            speed = 3  # good palette quality, 3-5x faster than speed=1
        else:
            max_colors = 256
            speed = 4  # default balanced

        # Lossy path: run pngquant first, then oxipng on the result
        pngquant_result, success = await self._run_pngquant(
            data_clean, config.quality, max_colors, speed
        )

        if success and pngquant_result:
            # Squeeze extra bytes from the lossy result
            lossy_optimized = await asyncio.to_thread(
                self._run_oxipng, pngquant_result, oxipng_level
            )
            # pngquant can inflate gradient/palette PNGs due to dithering;
            # fall through to oxipng-on-original when that happens.
            if len(lossy_optimized) <= len(data_clean):
                return self._build_result(data, lossy_optimized, "pngquant + oxipng")

        # pngquant failed, couldn't meet threshold, or produced bloat — lossless only
        optimized = await asyncio.to_thread(self._run_oxipng, data_clean, oxipng_level)
        return self._build_result(data, optimized, "oxipng")

    async def _run_pngquant(
        self,
        data: bytes,
        quality: int,
        max_colors: int = 256,
        speed: int = 4,
    ) -> tuple[bytes | None, bool]:
        """Run pngquant with quality-dependent settings.

        Uses floor=1 so pngquant always succeeds (never exit 99).
        Max colors varies by quality: 128 for aggressive, 256 for moderate.
        Speed: 1=slowest/best palette, 4=default, 11=fastest/roughest.

        Returns:
            (output_bytes, success). success=False when exit code 99
            (quality threshold cannot be met).
        """
        cmd = [
            "pngquant",
            str(max_colors),
            "--quality",
            f"1-{quality}",
            "--speed",
            str(speed),
            "-",
            "--output",
            "-",
        ]

        stdout, stderr, returncode = await run_tool(
            cmd,
            data,
            allowed_exit_codes={99},
        )

        if returncode == 99:
            return None, False

        return stdout, True

    def _run_oxipng(self, data: bytes, level: int = 2) -> bytes:
        """Run oxipng in-process via pyoxipng library (no subprocess).

        Level: 0=fastest/least compression, 6=slowest/best compression.
        """
        return oxipng.optimize_from_memory(data, level=level)
