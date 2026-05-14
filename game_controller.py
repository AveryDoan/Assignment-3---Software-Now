"""Application controller that coordinates GUI, game logic, and OpenCV.

Owner: Avery
Responsibility: architecture, integration, round setup, resets, difficulty
switching, hint flow, magnifier feed, timer ticks, heatmap reveal, and
best-score persistence to disk (JSON).
"""

from __future__ import annotations

import json
from pathlib import Path
import tkinter as tk

import numpy as np

from game_logic import ClickStatus, DifferenceRegion, GameLogic
from gui import SpotDifferenceGUI
from image_processor import (
    Difficulty,
    FOUND_COLOR,
    HINT_COLOR,
    ImageProcessor,
    REVEAL_COLOR,
)


SCORES_FILE = Path("scores.json")


class GameController:
    """Coordinates all layers of the Spot the Difference application."""

    display_max_width = 520
    display_max_height = 520
    magnifier_size = 80
    magnifier_zoom = 2.0

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self._difficulty = Difficulty.medium()

        self.processor = ImageProcessor(difficulty=self._difficulty)
        self.logic = GameLogic(
            max_mistakes=self._difficulty.max_mistakes,
            click_tolerance=self._difficulty.click_tolerance,
            base_score_per_find=self._difficulty.base_score,
            time_bonus_threshold=self._difficulty.time_bonus_threshold,
        )
        self.gui = SpotDifferenceGUI(root, self)

        self._current_path: Path | None = None
        self._base_original: np.ndarray | None = None
        self._base_modified: np.ndarray | None = None
        self._display_scale = 1.0
        self._total_score = 0
        self._best_score = self._load_best_score()
        self._summary_shown = False

        self.gui.set_timer_callback(self._on_timer_tick)
        self.gui.start_timer_loop()
        self.gui.set_status("Load a JPG, PNG, or BMP image to start.")
        self._sync_gui_state()

    def run(self) -> None:
        self.root.mainloop()

    # ---- top-level actions --------------------------------------------------

    def load_image_from_dialog(self) -> None:
        path = self.gui.ask_image_path()
        if not path:
            return
        self._start_round(Path(path))

    def reset_current_image(self) -> None:
        if self._current_path is None:
            self.gui.show_info("No image loaded", "Load an image before resetting.")
            return
        self._start_round(self._current_path)

    def reveal_differences(self) -> None:
        if self._base_original is None or self._base_modified is None:
            self.gui.show_info("No image loaded", "Load an image before revealing.")
            return

        self.logic.reveal_unfound()
        self._render_current_images(show_heatmap=True)
        self._sync_gui_state()
        self.gui.set_status("Unfound differences revealed. Colour = time-to-find.")
        if not self._summary_shown:
            self._summary_shown = True
            self._show_round_summary("Differences revealed")

    def use_hint(self) -> None:
        if not self.logic.can_use_hint:
            self.gui.set_status("Hints unavailable right now.")
            return
        hint_region = self.logic.request_hint()
        if hint_region is None:
            self.gui.set_status("No more hints available.")
            return
        self._render_current_images()
        self._sync_gui_state()
        self.gui.set_status(
            f"Hint: a difference is in the highlighted quadrant. "
            f"({self.logic.mistakes}/{self.logic.max_mistakes} mistakes)"
        )

    def set_difficulty(self, name: str) -> None:
        mapping = {"Easy": Difficulty.easy(), "Medium": Difficulty.medium(),
                   "Hard": Difficulty.hard()}
        new_difficulty = mapping.get(name, Difficulty.medium())
        self._difficulty = new_difficulty
        self.processor.set_difficulty(new_difficulty)
        self.logic.configure(
            max_mistakes=new_difficulty.max_mistakes,
            click_tolerance=new_difficulty.click_tolerance,
            base_score_per_find=new_difficulty.base_score,
            time_bonus_threshold=new_difficulty.time_bonus_threshold,
        )
        self.gui.set_status(
            f"Difficulty set to {new_difficulty.name}. "
            f"Load a new image to apply."
        )
        self._sync_gui_state()

    # ---- click + motion -----------------------------------------------------

    def handle_modified_click(self, display_x: int, display_y: int) -> None:
        if self._base_original is None or self._base_modified is None:
            self.gui.set_status("Load an image before guessing.")
            return

        original_x = int(display_x / self._display_scale)
        original_y = int(display_y / self._display_scale)
        result = self.logic.handle_click(original_x, original_y)

        # Visual feedback at the click location (display-space coords)
        if result.status in (ClickStatus.FOUND, ClickStatus.COMPLETE):
            self.gui.flash_click(display_x, display_y, success=True)
            self._total_score += 1
        elif result.status == ClickStatus.MISS or (
            result.status == ClickStatus.LOCKED and not self.logic.is_complete
        ):
            self.gui.flash_click(display_x, display_y, success=False)

        self._render_current_images(show_heatmap=self.logic.is_complete)
        self._sync_gui_state()
        self.gui.set_status(result.message)

        if result.status == ClickStatus.COMPLETE:
            self._handle_round_finish("Round complete!")
        elif result.status == ClickStatus.LOCKED and self.logic.is_locked and not self.logic.is_complete:
            self._handle_round_finish("Too many mistakes")

    def handle_magnifier_motion(self, display_x: int, display_y: int) -> None:
        """Render a zoomed crop of the modified image at the cursor."""
        if self._base_modified is None:
            return
        # Convert display coords back to original coords for the crop
        ox = int(display_x / self._display_scale)
        oy = int(display_y / self._display_scale)
        # Use the *marked* image so the magnifier shows reds/blues too
        marked = self._build_marked_modified()
        zoomed = self.processor.crop_zoom(
            marked, ox, oy,
            size=self.magnifier_size, zoom=self.magnifier_zoom,
        )
        self.gui.update_magnifier(zoomed)

    # ---- private helpers ----------------------------------------------------

    def _on_timer_tick(self) -> None:
        if self._base_original is None:
            return
        self.gui.update_timer(self.logic.elapsed_seconds)

    def _start_round(self, image_path: Path) -> None:
        try:
            processed = self.processor.load_and_generate(image_path)
        except (FileNotFoundError, ValueError) as exc:
            self.gui.show_error("Image load failed", str(exc))
            return
        except OSError as exc:
            self.gui.show_error("Image load failed", f"Could not read file: {exc}")
            return

        self._current_path = image_path
        self._base_original = processed.original
        self._base_modified = processed.modified
        self.logic.start_round(processed.regions)
        self._summary_shown = False
        self._render_current_images()
        self._sync_gui_state()
        self.gui.set_status(
            f"New round loaded: {image_path.name}  "
            f"(difficulty: {self._difficulty.name})"
        )

    def _build_marked_modified(self) -> np.ndarray:
        """Return modified image with the same markers the GUI currently shows."""
        assert self._base_modified is not None
        img = self.processor.mark_regions(
            self._base_modified, self.logic.found_regions, FOUND_COLOR
        )
        img = self.processor.mark_regions(img, self.logic.revealed_regions, REVEAL_COLOR)
        for region in self.logic.hinted_regions:
            if region not in self.logic.found_regions:
                img = self.processor.draw_hint_quadrant(img, region)
        return img

    def _render_current_images(self, show_heatmap: bool = False) -> None:
        if self._base_original is None or self._base_modified is None:
            return

        original = self._base_original.copy()
        modified = self._base_modified.copy()

        # Hint quadrants (drawn first so circles sit on top)
        for region in self.logic.hinted_regions:
            if region in self.logic.found_regions:
                continue
            original = self.processor.draw_hint_quadrant(original, region)
            modified = self.processor.draw_hint_quadrant(modified, region)

        # Found = red circles on both
        original = self.processor.mark_regions(original, self.logic.found_regions, FOUND_COLOR)
        modified = self.processor.mark_regions(modified, self.logic.found_regions, FOUND_COLOR)

        # Revealed differences: heatmap colours after round end, plain blue otherwise
        if show_heatmap and (self.logic.is_complete or self.logic.is_revealed or self.logic.is_locked):
            summary = self.logic.build_summary()
            original = self.processor.draw_heatmap_markers(original, list(summary.timings))
            modified = self.processor.draw_heatmap_markers(modified, list(summary.timings))
        else:
            original = self.processor.mark_regions(
                original, self.logic.revealed_regions, REVEAL_COLOR
            )
            modified = self.processor.mark_regions(
                modified, self.logic.revealed_regions, REVEAL_COLOR
            )

        original_scaled = self.processor.resize_to_fit(
            original, self.display_max_width, self.display_max_height
        )
        modified_scaled = self.processor.resize_to_fit(
            modified, self.display_max_width, self.display_max_height
        )

        self._display_scale = modified_scaled.scale
        self.gui.render_images(original_scaled.image, modified_scaled.image)

    def _sync_gui_state(self) -> None:
        has_round = self._base_original is not None
        self.gui.update_counters(
            remaining=self.logic.remaining_count,
            found=self.logic.found_count,
            total=self.logic.total_count,
            mistakes=self.logic.mistakes,
            max_mistakes=self.logic.max_mistakes,
            hints_used=self.logic.hints_used,
            total_score=self._total_score,
            best_score=self._best_score,
            has_round=has_round,
        )
        self.gui.set_clicks_enabled(self.logic.can_accept_clicks)
        self.gui.set_reveal_enabled(has_round and self.logic.remaining_count > 0)
        self.gui.set_reset_enabled(has_round)
        self.gui.set_hint_enabled(self.logic.can_use_hint)
        self._sync_banner(has_round)

    def _sync_banner(self, has_round: bool) -> None:
        if not has_round:
            self.gui.set_banner("")
            return
        if self.logic.is_complete:
            summary = self.logic.build_summary()
            self.gui.set_banner(
                f"All {self.logic.total_count} found in "
                f"{summary.elapsed_seconds:.1f}s — round score {summary.score}",
                mode="success",
            )
        elif self.logic.is_locked:
            self.gui.set_banner(
                f"Too many incorrect guesses ({self.logic.max_mistakes}/"
                f"{self.logic.max_mistakes}). Found "
                f"{self.logic.found_count}/{self.logic.total_count}. "
                "Load a new image to restart.",
                mode="error",
            )
        elif self.logic.is_revealed:
            self.gui.set_banner(
                "Unfound differences revealed. Colour = time-to-find.",
                mode="info",
            )
        else:
            self.gui.set_banner("")

    def _handle_round_finish(self, headline: str) -> None:
        if self._summary_shown:
            return
        self._summary_shown = True
        summary = self.logic.build_summary()
        # Update best score on completed rounds only
        if self.logic.is_complete and summary.score > self._best_score:
            self._best_score = summary.score
            self._save_best_score(self._best_score)
        self._show_round_summary(headline)
        self._sync_gui_state()

    def _show_round_summary(self, headline: str) -> None:
        summary = self.logic.build_summary()
        body = (
            f"{headline}\n\n"
            f"Found: {summary.found}/{summary.total}\n"
            f"Mistakes: {summary.mistakes}\n"
            f"Hints used: {summary.hints_used}\n"
            f"Time: {summary.elapsed_seconds:.1f}s\n"
            f"Round score: {summary.score}\n"
            f"Best so far: {self._best_score}\n\n"
            "Load or reset an image to play again."
        )
        self.gui.show_round_summary("Round summary", body)

    # ---- persistence --------------------------------------------------------

    def _load_best_score(self) -> int:
        try:
            with SCORES_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return int(data.get("best_score", 0))
        except (FileNotFoundError, ValueError, OSError):
            return 0

    def _save_best_score(self, value: int) -> None:
        try:
            with SCORES_FILE.open("w", encoding="utf-8") as fh:
                json.dump({"best_score": int(value)}, fh)
        except OSError:
            pass  # non-fatal — game still works without persistence
