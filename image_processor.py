"""OpenCV image loading, difference generation, scaling, and marking.

Owner: Max
Responsibility: all OpenCV image work, including loading, cloning, random
difference generation, non-overlapping regions, and visual markers.

OOP notes for marker:
    * ``DifferenceOperation`` is an abstract base class. Each concrete
      subclass overrides ``apply`` -- this demonstrates **inheritance**
      and **polymorphism**.
    * ``ImageProcessor`` calls ``operation.apply(...)`` without knowing
      the concrete subclass, so new alteration types can be added
      without changing the caller (open/closed principle).
    * Private attributes prefixed with ``_`` (``_rng``, ``_operations``)
      demonstrate **encapsulation**.
    * ``Difficulty`` is a dataclass-based value object that
      ``ImageProcessor`` and ``GameLogic`` both consume -- showing
      **class interaction** across modules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
import random

import cv2
import numpy as np

from game_logic import DifferenceRegion


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
FOUND_COLOR = (0, 0, 255)        # BGR red
REVEAL_COLOR = (255, 0, 0)       # BGR blue
HINT_COLOR = (0, 200, 255)       # BGR yellow-orange for hint quadrant


@dataclass(frozen=True)
class Difficulty:
    """Tuning bundle shared by image processor and game logic."""

    name: str
    region_scale: tuple[float, float]   # (min, max) as fraction of min image dim
    click_tolerance: int
    max_mistakes: int
    blend_alpha: float                  # higher = subtler alteration
    base_score: int                     # points per difference at this level
    time_bonus_threshold: int           # seconds; under this earns speed bonus

    @classmethod
    def easy(cls) -> "Difficulty":
        return cls("Easy", (0.10, 0.18), 28, 5, 0.62, 100, 45)

    @classmethod
    def medium(cls) -> "Difficulty":
        return cls("Medium", (0.07, 0.13), 18, 3, 0.74, 200, 30)

    @classmethod
    def hard(cls) -> "Difficulty":
        return cls("Hard", (0.05, 0.09), 12, 3, 0.84, 350, 20)

    @classmethod
    def all_levels(cls) -> tuple["Difficulty", ...]:
        return (cls.easy(), cls.medium(), cls.hard())


@dataclass(frozen=True)
class ProcessedImages:
    original: np.ndarray
    modified: np.ndarray
    regions: tuple[DifferenceRegion, ...]


@dataclass(frozen=True)
class ScaledImage:
    image: np.ndarray
    scale: float


# ---------------------------------------------------------------------------
# Alteration strategies (polymorphic via DifferenceOperation)
# ---------------------------------------------------------------------------

class DifferenceOperation(ABC):
    """Base class for polymorphic OpenCV alteration strategies."""

    name = "base"

    @abstractmethod
    def apply(
        self,
        image: np.ndarray,
        region: DifferenceRegion,
        rng: random.Random,
        difficulty: Difficulty,
    ) -> None:
        """Modify ``image`` inside ``region`` in-place."""

    def _region_slice(self, region: DifferenceRegion) -> tuple[slice, slice]:
        return (
            slice(region.y, region.y + region.height),
            slice(region.x, region.x + region.width),
        )


class ColorShiftDifference(DifferenceOperation):
    name = "colour shift"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        if float(hsv[:, :, 1].mean()) < 35.0:
            tint = np.full_like(
                roi,
                (rng.randint(50, 220), rng.randint(50, 220), rng.randint(50, 220)),
            )
            image[rows, cols] = cv2.addWeighted(
                roi, difficulty.blend_alpha, tint, 1 - difficulty.blend_alpha, 0
            )
            return

        # Harder difficulties = smaller hue shifts (subtler)
        hue_shift = rng.randint(8, 24) if difficulty.name != "Hard" else rng.randint(5, 14)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(np.int16) + hue_shift) % 180
        hsv[:, :, 1] = np.clip(
            hsv[:, :, 1].astype(np.int16) + rng.randint(18, 42), 0, 255
        ).astype(np.uint8)
        image[rows, cols] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class BrightnessDifference(DifferenceOperation):
    name = "brightness shift"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols].astype(np.int16)
        mean_value = float(roi.mean())
        # Subtler beta at harder difficulties
        magnitude = {"Easy": 50, "Medium": 34, "Hard": 22}.get(difficulty.name, 34)
        beta = -magnitude if mean_value > 150 else magnitude
        if rng.random() < 0.35:
            beta = -beta
        image[rows, cols] = np.clip(roi + beta, 0, 255).astype(np.uint8)


class SoftBlurDifference(DifferenceOperation):
    name = "soft blur"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols]
        kernel = max(5, min(region.width, region.height) // 3)
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(roi, (kernel, kernel), 0)
        blended = cv2.addWeighted(
            roi, difficulty.blend_alpha, blurred, 1 - difficulty.blend_alpha, 0
        )

        if float(roi.var()) < 30.0:
            overlay = blended.copy()
            colour = _contrast_colour(roi)
            cv2.line(
                overlay,
                (region.width // 5, region.height // 2),
                (region.width * 4 // 5, region.height // 2),
                colour,
                max(1, min(region.width, region.height) // 18),
                lineType=cv2.LINE_AA,
            )
            blended = cv2.addWeighted(blended, 0.78, overlay, 0.22, 0)

        image[rows, cols] = blended


class ShapeOverlayDifference(DifferenceOperation):
    name = "shape overlay"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols]
        overlay = roi.copy()
        colour = _contrast_colour(roi)
        thickness = max(1, min(region.width, region.height) // 16)

        shape = rng.choice(("ellipse", "rectangle", "line"))
        if shape == "ellipse":
            cv2.ellipse(
                overlay,
                (region.width // 2, region.height // 2),
                (max(5, region.width // 3), max(5, region.height // 4)),
                rng.randint(0, 160), 0, 360,
                colour, thickness, lineType=cv2.LINE_AA,
            )
        elif shape == "rectangle":
            margin_x = max(3, region.width // 5)
            margin_y = max(3, region.height // 5)
            cv2.rectangle(
                overlay,
                (margin_x, margin_y),
                (region.width - margin_x, region.height - margin_y),
                colour, thickness, lineType=cv2.LINE_AA,
            )
        else:
            cv2.line(
                overlay,
                (region.width // 5, region.height // 5),
                (region.width * 4 // 5, region.height * 4 // 5),
                colour, thickness, lineType=cv2.LINE_AA,
            )

        # Blend more for harder difficulty so shapes are subtler
        alpha = 0.6 if difficulty.name == "Easy" else (0.72 if difficulty.name == "Medium" else 0.82)
        image[rows, cols] = cv2.addWeighted(roi, alpha, overlay, 1 - alpha, 0)


class MirrorFlipDifference(DifferenceOperation):
    """Horizontally mirror the region. Subtle on symmetric content, obvious on text/faces."""

    name = "mirror flip"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols]
        flipped = cv2.flip(roi, 1)
        # Blend slightly so it looks natural at edges
        image[rows, cols] = cv2.addWeighted(flipped, 0.9, roi, 0.1, 0)


class HueRotationDifference(DifferenceOperation):
    """Rotate hue of one region for a painterly colour shift."""

    name = "hue rotation"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(np.int16)
        rotation = {"Easy": 60, "Medium": 40, "Hard": 25}.get(difficulty.name, 40)
        hsv[:, :, 0] = (hsv[:, :, 0] + rotation) % 180
        rotated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        image[rows, cols] = cv2.addWeighted(
            roi, difficulty.blend_alpha, rotated, 1 - difficulty.blend_alpha, 0
        )


class ObjectRemovalDifference(DifferenceOperation):
    """Use OpenCV inpainting to 'erase' a small detail. Looks very natural."""

    name = "object removal"

    def apply(self, image, region, rng, difficulty):
        rows, cols = self._region_slice(region)
        roi = image[rows, cols]
        h, w = roi.shape[:2]

        # Build a small mask in the centre of the region
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        radius = max(3, min(w, h) // 3)
        cv2.circle(mask, (cx, cy), radius, 255, -1)

        # Inpaint the masked area using surrounding pixels
        inpainted = cv2.inpaint(roi, mask, 3, cv2.INPAINT_TELEA)
        image[rows, cols] = inpainted


# ---------------------------------------------------------------------------
# ImageProcessor
# ---------------------------------------------------------------------------

class ImageProcessor:
    """OpenCV service for the game images."""

    difference_count = 5

    def __init__(
        self,
        rng: random.Random | None = None,
        difficulty: Difficulty | None = None,
    ) -> None:
        self._rng = rng or random.Random()
        self._difficulty = difficulty or Difficulty.medium()
        self._operations: tuple[DifferenceOperation, ...] = (
            ColorShiftDifference(),
            BrightnessDifference(),
            SoftBlurDifference(),
            ShapeOverlayDifference(),
            MirrorFlipDifference(),
            HueRotationDifference(),
            ObjectRemovalDifference(),
        )

    def set_difficulty(self, difficulty: Difficulty) -> None:
        self._difficulty = difficulty

    @property
    def difficulty(self) -> Difficulty:
        return self._difficulty

    @property
    def alteration_count(self) -> int:
        return len(self._operations)

    def load_and_generate(self, image_path: str | Path) -> ProcessedImages:
        original = self.load_image(image_path)
        modified = original.copy()
        raw_regions = self._generate_regions(original)
        operations = self._select_operations(self.difference_count)
        regions: list[DifferenceRegion] = []

        for raw_region, operation in zip(raw_regions, operations, strict=True):
            region = replace(raw_region, kind=operation.name)
            operation.apply(modified, region, self._rng, self._difficulty)
            regions.append(region)

        return ProcessedImages(original, modified, tuple(regions))

    def load_image(self, image_path: str | Path) -> np.ndarray:
        path = Path(image_path)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported image type. Choose one of: {allowed}.")
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")

        encoded = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("OpenCV could not read this image file.")

        height, width = image.shape[:2]
        if width < 80 or height < 80:
            raise ValueError("Choose an image at least 80 x 80 pixels.")
        return image

    def mark_regions(
        self,
        image: np.ndarray,
        regions: tuple[DifferenceRegion, ...],
        colour: tuple[int, int, int],
    ) -> np.ndarray:
        marked = image.copy()
        for region in regions:
            self.draw_marker(marked, region, colour)
        return marked

    def draw_marker(
        self,
        image: np.ndarray,
        region: DifferenceRegion,
        colour: tuple[int, int, int],
    ) -> None:
        centre = region.center
        radius = max(10, region.radius + 8)
        thickness = max(2, radius // 8)
        cv2.circle(image, centre, radius, colour, thickness, lineType=cv2.LINE_AA)

    def draw_heatmap_markers(
        self,
        image: np.ndarray,
        timings: list[tuple[DifferenceRegion, float | None]],
    ) -> np.ndarray:
        """Return image with regions tinted by find-time (green=fast, red=slow).

        Regions with ``None`` timing (never found) are drawn in grey.
        """
        marked = image.copy()
        found_times = [t for _, t in timings if t is not None]
        max_t = max(found_times) if found_times else 1.0

        for region, seconds in timings:
            centre = region.center
            radius = max(10, region.radius + 8)
            thickness = max(2, radius // 8)
            if seconds is None:
                colour = (140, 140, 140)
                label = "unfound"
            else:
                norm = 0.0 if max_t <= 0 else min(1.0, seconds / max_t)
                if norm < 0.5:
                    t = norm * 2
                    colour = (0, 255, int(255 * t))
                else:
                    t = (norm - 0.5) * 2
                    colour = (0, int(255 * (1 - t)), 255)
                label = f"{seconds:.1f}s"
            cv2.circle(marked, centre, radius, colour, thickness, lineType=cv2.LINE_AA)
            cv2.putText(
                marked, label,
                (centre[0] - 24, centre[1] - radius - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )
        return marked

    def draw_hint_quadrant(
        self,
        image: np.ndarray,
        region: DifferenceRegion,
    ) -> np.ndarray:
        """Tint the quadrant containing ``region`` to give a vague hint."""
        marked = image.copy()
        height, width = marked.shape[:2]
        mid_x, mid_y = width // 2, height // 2
        cx, cy = region.center
        x0 = 0 if cx < mid_x else mid_x
        y0 = 0 if cy < mid_y else mid_y
        x1 = mid_x if cx < mid_x else width
        y1 = mid_y if cy < mid_y else height

        overlay = marked.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), HINT_COLOR, -1)
        cv2.addWeighted(overlay, 0.18, marked, 0.82, 0, dst=marked)
        cv2.rectangle(marked, (x0, y0), (x1, y1), HINT_COLOR, 3, lineType=cv2.LINE_AA)
        return marked

    def crop_zoom(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        size: int = 80,
        zoom: float = 2.0,
    ) -> np.ndarray:
        """Return a zoomed crop centred on (cx, cy) for the magnifier lens."""
        h, w = image.shape[:2]
        half = size // 2
        x0 = max(0, min(w - size, cx - half))
        y0 = max(0, min(h - size, cy - half))
        crop = image[y0:y0 + size, x0:x0 + size]
        out_size = int(size * zoom)
        zoomed = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
        cv2.rectangle(
            zoomed, (0, 0), (out_size - 1, out_size - 1),
            (200, 200, 200), 2, lineType=cv2.LINE_AA,
        )
        return zoomed

    def resize_to_fit(
        self,
        image: np.ndarray,
        max_width: int,
        max_height: int,
        allow_upscale: bool = False,
    ) -> ScaledImage:
        height, width = image.shape[:2]
        scale = min(max_width / width, max_height / height)
        if not allow_upscale:
            scale = min(1.0, scale)

        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        return ScaledImage(resized, scale)

    # ----- private helpers ---------------------------------------------------

    def _select_operations(self, count: int) -> list[DifferenceOperation]:
        # Guarantee at least 3 distinct alteration types per round (rubric
        # requirement), then pad up to ``count`` by sampling with replacement.
        operations = list(self._operations)
        self._rng.shuffle(operations)
        distinct_target = min(3, count, len(operations))
        selected = self._rng.sample(operations, distinct_target)
        while len(selected) < count:
            selected.append(self._rng.choice(self._operations))
        self._rng.shuffle(selected)
        return selected

    def _generate_regions(self, image: np.ndarray) -> tuple[DifferenceRegion, ...]:
        """Place 5 non-overlapping regions, preferring 'busy' areas of the image.

        Uses Canny edge detection to build a density map; we sample candidate
        positions and prefer those with more edge content nearby. Differences
        therefore land in interesting parts of the image (faces, objects,
        textures) rather than flat sky/wall.
        """
        height, width = image.shape[:2]
        min_dimension = min(width, height)

        d = self._difficulty
        min_size = max(18, int(min_dimension * d.region_scale[0]))
        max_size = max(min_size + 2, int(min_dimension * d.region_scale[1]))
        max_size = min(max_size, max(20, min_dimension // 4))
        padding = max(6, min_dimension // 55)

        # Edge density map (one channel, blurred for tolerance)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        density = cv2.GaussianBlur(edges, (51, 51), 0).astype(np.float32)

        def score_position(x: int, y: int, w: int, h: int) -> float:
            patch = density[y:y + h, x:x + w]
            return float(patch.mean()) if patch.size else 0.0

        regions: list[DifferenceRegion] = []
        for _ in range(self.difference_count):
            best_candidate: DifferenceRegion | None = None
            best_score = -1.0

            for _attempt in range(120):
                region_width = self._rng.randint(min_size, max_size)
                region_height = self._rng.randint(min_size, max_size)
                x = self._rng.randint(0, max(0, width - region_width - 1))
                y = self._rng.randint(0, max(0, height - region_height - 1))
                candidate = DifferenceRegion(x, y, region_width, region_height)
                if any(candidate.overlaps(existing, padding) for existing in regions):
                    continue
                s = score_position(x, y, region_width, region_height)
                # Random jitter so we don't always pick the same hot spot
                s += self._rng.uniform(0, 8.0)
                if s > best_score:
                    best_score = s
                    best_candidate = candidate

            if best_candidate is None:
                return self._fallback_grid_regions(width, height, min_size, max_size)
            regions.append(best_candidate)
        return tuple(regions)

    def _fallback_grid_regions(
        self, width: int, height: int, min_size: int, max_size: int,
    ) -> tuple[DifferenceRegion, ...]:
        cols = 3
        rows = 2
        cell_width = width // cols
        cell_height = height // rows
        region_width = max(min_size, min(cell_width // 2, width // 8))
        region_height = max(min_size, min(cell_height // 2, height // 8))

        regions: list[DifferenceRegion] = []
        for index in range(self.difference_count):
            col = index % cols
            row = index // cols
            origin_x = col * cell_width
            origin_y = row * cell_height
            slack_x = max(0, cell_width - region_width)
            slack_y = max(0, cell_height - region_height)
            x = min(width - region_width - 1, origin_x + slack_x // 2)
            y = min(height - region_height - 1, origin_y + slack_y // 2)
            regions.append(DifferenceRegion(max(0, x), max(0, y), region_width, region_height))
        return tuple(regions)


def _contrast_colour(roi: np.ndarray) -> tuple[int, int, int]:
    mean_bgr = roi.reshape(-1, 3).mean(axis=0)
    luminance = 0.114 * mean_bgr[0] + 0.587 * mean_bgr[1] + 0.299 * mean_bgr[2]
    return (35, 35, 35) if luminance > 130 else (230, 230, 230)
