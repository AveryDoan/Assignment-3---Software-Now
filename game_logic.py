"""Round rules and click validation for the Spot the Difference game.

Owner: Alvi
Responsibility: click detection, mistakes, scoring, reveal state, timer,
per-difference find-time tracking, hint accounting, and reset behaviour for
one image round.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Iterable


@dataclass(frozen=True)
class DifferenceRegion:
    """A rectangular difference area in original image coordinates."""

    x: int
    y: int
    width: int
    height: int
    kind: str = "unknown"

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    @property
    def radius(self) -> int:
        return max(self.width, self.height) // 2

    def contains(self, click_x: int, click_y: int, tolerance: int = 0) -> bool:
        return (
            self.x - tolerance <= click_x <= self.x + self.width + tolerance
            and self.y - tolerance <= click_y <= self.y + self.height + tolerance
        )

    def overlaps(self, other: "DifferenceRegion", padding: int = 0) -> bool:
        return not (
            self.x + self.width + padding < other.x
            or other.x + other.width + padding < self.x
            or self.y + self.height + padding < other.y
            or other.y + other.height + padding < self.y
        )


class ClickStatus(str, Enum):
    """Possible outcomes after the player clicks the modified image."""

    FOUND = "found"
    COMPLETE = "complete"
    MISS = "miss"
    ALREADY_MARKED = "already_marked"
    LOCKED = "locked"


@dataclass(frozen=True)
class ClickResult:
    status: ClickStatus
    message: str
    region_index: int | None = None
    seconds_to_find: float | None = None

    @property
    def is_success(self) -> bool:
        return self.status in {ClickStatus.FOUND, ClickStatus.COMPLETE}


@dataclass(frozen=True)
class RoundSummary:
    """Snapshot of a finished round, used for scoring and the heatmap."""

    found: int
    total: int
    mistakes: int
    elapsed_seconds: float
    hints_used: int
    score: int
    timings: tuple[tuple[DifferenceRegion, float | None], ...]


class GameLogic:
    """Keeps all mutable game-rule state for the current image."""

    def __init__(
        self,
        max_mistakes: int = 3,
        click_tolerance: int = 18,
        base_score_per_find: int = 200,
        time_bonus_threshold: int = 30,
    ) -> None:
        self.max_mistakes = max_mistakes
        self.click_tolerance = click_tolerance
        self.base_score_per_find = base_score_per_find
        self.time_bonus_threshold = time_bonus_threshold

        self._regions: list[DifferenceRegion] = []
        self._found_indices: set[int] = set()
        self._revealed_indices: set[int] = set()
        self._hinted_indices: set[int] = set()
        self._mistakes = 0
        self._hints_used = 0
        self._round_start: float | None = None
        self._round_end: float | None = None
        self._find_times: dict[int, float] = {}  # region_index -> seconds_to_find

    # configuration 

    def configure(
        self,
        max_mistakes: int,
        click_tolerance: int,
        base_score_per_find: int,
        time_bonus_threshold: int,
    ) -> None:
        self.max_mistakes = max_mistakes
        self.click_tolerance = click_tolerance
        self.base_score_per_find = base_score_per_find
        self.time_bonus_threshold = time_bonus_threshold

    def start_round(self, regions: Iterable[DifferenceRegion]) -> None:
        self._regions = list(regions)
        self._found_indices.clear()
        self._revealed_indices.clear()
        self._hinted_indices.clear()
        self._mistakes = 0
        self._hints_used = 0
        self._round_start = time.monotonic()
        self._round_end = None
        self._find_times.clear()

    # read-only state 

    @property
    def regions(self) -> tuple[DifferenceRegion, ...]:
        return tuple(self._regions)

    @property
    def mistakes(self) -> int:
        return self._mistakes

    @property
    def hints_used(self) -> int:
        return self._hints_used

    @property
    def found_count(self) -> int:
        return len(self._found_indices)

    @property
    def total_count(self) -> int:
        return len(self._regions)

    @property
    def remaining_count(self) -> int:
        resolved = self._found_indices | self._revealed_indices
        return max(0, len(self._regions) - len(resolved))

    @property
    def is_complete(self) -> bool:
        return bool(self._regions) and len(self._found_indices) == len(self._regions)

    @property
    def is_revealed(self) -> bool:
        return bool(self._revealed_indices)

    @property
    def is_locked(self) -> bool:
        return self._mistakes >= self.max_mistakes

    @property
    def can_accept_clicks(self) -> bool:
        return bool(self._regions) and not (
            self.is_complete or self.is_revealed or self.is_locked
        )

    @property
    def can_use_hint(self) -> bool:
        # Hint costs 1 mistake, so it can't be used if that would lock the round.
        return (
            self.can_accept_clicks
            and self.remaining_count > 0
            and self._mistakes < self.max_mistakes - 1
        )

    @property
    def found_regions(self) -> tuple[DifferenceRegion, ...]:
        return tuple(self._regions[index] for index in sorted(self._found_indices))

    @property
    def revealed_regions(self) -> tuple[DifferenceRegion, ...]:
        return tuple(self._regions[index] for index in sorted(self._revealed_indices))

    @property
    def hinted_regions(self) -> tuple[DifferenceRegion, ...]:
        return tuple(self._regions[index] for index in sorted(self._hinted_indices))

    @property
    def elapsed_seconds(self) -> float:
        if self._round_start is None:
            return 0.0
        end = self._round_end or time.monotonic()
        return max(0.0, end - self._round_start)

    # click handling 

    def handle_click(self, click_x: int, click_y: int) -> ClickResult:
        if not self._regions:
            return ClickResult(ClickStatus.LOCKED, "Load an image to start a round.")

        if not self.can_accept_clicks:
            return ClickResult(ClickStatus.LOCKED, self._locked_message())

        already_marked = self._find_region(
            click_x, click_y, self._found_indices | self._revealed_indices
        )
        if already_marked is not None:
            return ClickResult(
                ClickStatus.ALREADY_MARKED,
                "That difference is already marked.",
                already_marked,
            )

        match_index = self._find_region(
            click_x,
            click_y,
            set(range(len(self._regions))) - self._found_indices - self._revealed_indices,
        )
        if match_index is not None:
            seconds = self.elapsed_seconds
            self._found_indices.add(match_index)
            self._find_times[match_index] = seconds
            if self.is_complete:
                self._round_end = time.monotonic()
                return ClickResult(
                    ClickStatus.COMPLETE,
                    "All differences found.",
                    match_index,
                    seconds,
                )
            return ClickResult(ClickStatus.FOUND, "Difference found.", match_index, seconds)

        self._mistakes += 1
        if self.is_locked:
            self._round_end = time.monotonic()
            return ClickResult(ClickStatus.LOCKED, self._locked_message())
        return ClickResult(ClickStatus.MISS, "No difference at that position.")

    def request_hint(self) -> DifferenceRegion | None:
        """Pick an unfound, un-hinted region. Costs one mistake."""
        if not self.can_use_hint:
            return None
        candidates = (
            set(range(len(self._regions)))
            - self._found_indices
            - self._revealed_indices
            - self._hinted_indices
        )
        if not candidates:
            # Recycle hints if all unfound regions have already been hinted
            candidates = (
                set(range(len(self._regions)))
                - self._found_indices
                - self._revealed_indices
            )
        if not candidates:
            return None
        index = min(candidates)  # deterministic so hint button feels predictable
        self._hinted_indices.add(index)
        self._mistakes += 1
        self._hints_used += 1
        return self._regions[index]

    def reveal_unfound(self) -> tuple[DifferenceRegion, ...]:
        unfound = set(range(len(self._regions))) - self._found_indices
        self._revealed_indices = unfound
        if self._round_end is None:
            self._round_end = time.monotonic()
        return self.revealed_regions

    # scoring 

    def compute_round_score(self) -> int:
        """Score this round: finds * base − mistakes * 50 − hints * 75
        + speed bonus if all found under threshold.
        """
        score = self.found_count * self.base_score_per_find
        score -= self._mistakes * 50
        score -= self._hints_used * 75
        if self.is_complete and self.elapsed_seconds <= self.time_bonus_threshold:
            score += 250
        return max(0, score)

    def build_summary(self) -> RoundSummary:
        timings: list[tuple[DifferenceRegion, float | None]] = []
        for index, region in enumerate(self._regions):
            timings.append((region, self._find_times.get(index)))
        return RoundSummary(
            found=self.found_count,
            total=self.total_count,
            mistakes=self._mistakes,
            elapsed_seconds=self.elapsed_seconds,
            hints_used=self._hints_used,
            score=self.compute_round_score(),
            timings=tuple(timings),
        )

    # private 

    def _find_region(
        self, click_x: int, click_y: int, candidates: Iterable[int]
    ) -> int | None:
        for index in candidates:
            if self._regions[index].contains(click_x, click_y, self.click_tolerance):
                return index
        return None

    def _locked_message(self) -> str:
        # Order matters: report the most informative condition first.
        if self.is_locked and not self.is_complete:
            return (
                f"Too many incorrect guesses! You found {self.found_count} "
                f"of {self.total_count} differences. Load a new image to restart."
            )
        if self.is_complete:
            return "Round complete. Load or reset an image to play again."
        if self.is_revealed:
            return "Differences revealed. Load or reset an image to play again."
        return "Clicks are currently disabled."
