import asyncio
import io

from PIL import Image

from optimizers.base import BaseOptimizer
from optimizers.utils import binary_search_quality
from schemas import OptimizationConfig, OptimizeResult
from utils.format_detect import ImageFormat


class WebpOptimizer(BaseOptimizer):
    """WebP optimization: Pillow re-encode with method=4.

    Pipeline:
    1. Decode image once
    2. Re-encode with Pillow at target quality
    3. If max_reduction set and exceeded, binary search quality (reuses decoded img)
    """

    format = ImageFormat.WEBP

    async def optimize(self, data: bytes, config: OptimizationConfig) -> OptimizeResult:
        # Decode once, share across all paths
        img, is_animated = await asyncio.to_thread(self._decode_image, data)

        best = await asyncio.to_thread(self._encode_webp, img, config.quality, is_animated)
        method = "pillow"

        # Cap reduction if max_reduction is set (reuses pre-decoded img)
        if config.max_reduction is not None:
            reduction = (1 - len(best) / len(data)) * 100
            if reduction > config.max_reduction:
                capped = await asyncio.to_thread(
                    self._find_capped_quality, img, is_animated, data, config
                )
                if capped is not None:
                    best = capped
                    method = "pillow"

        return self._build_result(data, best, method)

    @staticmethod
    def _decode_image(data: bytes) -> tuple[Image.Image, bool]:
        """Decode WebP once. Returns (img, is_animated)."""
        img = Image.open(io.BytesIO(data))
        is_animated = getattr(img, "n_frames", 1) > 1
        return img, is_animated

    def _find_capped_quality(
        self,
        img: Image.Image,
        is_animated: bool,
        data: bytes,
        config: OptimizationConfig,
    ) -> bytes | None:
        """Binary search Pillow quality to cap reduction at max_reduction."""

        def encode_fn(quality: int) -> bytes:
            return self._encode_webp(img, quality, is_animated)

        return binary_search_quality(
            encode_fn, len(data), config.max_reduction, lo=config.quality, hi=100
        )

    @staticmethod
    def _encode_webp(img: Image.Image, quality: int, is_animated: bool) -> bytes:
        """Encode a Pillow Image to WebP bytes."""
        if is_animated:
            img.seek(0)  # Reset frame pointer before re-encode

        output = io.BytesIO()

        save_kwargs = {
            "format": "WEBP",
            "quality": quality,
            "method": 4,
        }

        if is_animated:
            save_kwargs["save_all"] = True
            save_kwargs["minimize_size"] = True

        img.save(output, **save_kwargs)
        return output.getvalue()
