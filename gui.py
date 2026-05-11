"""Tkinter user interface for the Spot the Difference game.

Owner: Jane
Responsibility: window layout, controls, dialogs, counters, banner, image
display, click binding, click feedback animation, magnifier lens, dark mode,
difficulty selector, and live timer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
import numpy as np

if TYPE_CHECKING:
    from game_controller import GameController


# Palettes for light and dark mode
LIGHT_THEME = {
    "bg": "#f4f4f6",
    "fg": "#1a1a1a",
    "canvas_bg": "#f3f3f3",
    "canvas_border": "#c8c8c8",
    "muted": "#555555",
}
DARK_THEME = {
    "bg": "#1e1f24",
    "fg": "#e8e8ee",
    "canvas_bg": "#2a2b31",
    "canvas_border": "#4a4b52",
    "muted": "#a8a8b2",
}


class SpotDifferenceGUI:
    """Tkinter view layer for the game."""

    def __init__(self, root: tk.Tk, controller: "GameController") -> None:
        self.root = root
        self.controller = controller
        self._original_photo: ImageTk.PhotoImage | None = None
        self._modified_photo: ImageTk.PhotoImage | None = None
        self._magnifier_photo: ImageTk.PhotoImage | None = None
        self._clicks_enabled = False
        self._theme = LIGHT_THEME
        self._dark_mode = False
        self._magnifier_enabled = False

        # State variables
        self.remaining_var = tk.StringVar(value="Remaining: -")
        self.found_var = tk.StringVar(value="Found: -")
        self.mistakes_var = tk.StringVar(value="Mistakes: -")
        self.hints_var = tk.StringVar(value="Hints: 0")
        self.score_var = tk.StringVar(value="Total score: 0")
        self.best_score_var = tk.StringVar(value="Best: 0")
        self.timer_var = tk.StringVar(value="Time: 0.0s")
        self.status_var = tk.StringVar(value="")
        self.banner_var = tk.StringVar(value="")
        self.difficulty_var = tk.StringVar(value="Medium")
        self.magnifier_var = tk.BooleanVar(value=False)
        self.dark_mode_var = tk.BooleanVar(value=False)

        # Callbacks set by controller
        self._timer_callback: Callable[[], None] | None = None

        self._build_window()
        self._apply_theme()

    # ---- public API ---------------------------------------------------------

    def ask_image_path(self) -> str:
        return filedialog.askopenfilename(
            title="Choose an image",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("Bitmap", "*.bmp"),
                ("All files", "*.*"),
            ),
        )

    def render_images(self, original_bgr: np.ndarray, modified_bgr: np.ndarray) -> None:
        self._modified_source = modified_bgr  # kept for magnifier
        self._original_photo = self._to_photo(original_bgr)
        self._modified_photo = self._to_photo(modified_bgr)
        self._draw_photo(self.original_canvas, self._original_photo)
        self._draw_photo(self.modified_canvas, self._modified_photo)

    def update_counters(
        self,
        remaining: int,
        found: int,
        total: int,
        mistakes: int,
        max_mistakes: int,
        hints_used: int,
        total_score: int,
        best_score: int,
        has_round: bool,
    ) -> None:
        if not has_round:
            self.remaining_var.set("Remaining: -")
            self.found_var.set("Found: -")
            self.mistakes_var.set("Mistakes: -")
            self.hints_var.set("Hints: -")
        else:
            self.remaining_var.set(f"Remaining: {remaining}")
            self.found_var.set(f"Found: {found}/{total}")
            self.mistakes_var.set(f"Mistakes: {mistakes}/{max_mistakes}")
            self.hints_var.set(f"Hints: {hints_used}")
        self.score_var.set(f"Total score: {total_score}")
        self.best_score_var.set(f"Best: {best_score}")

    def update_timer(self, elapsed_seconds: float) -> None:
        self.timer_var.set(f"Time: {elapsed_seconds:.1f}s")

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def set_banner(self, message: str, mode: str = "info") -> None:
        """Coloured banner above the images. Empty message hides it."""
        # Palettes mirror semantic colours and contrast well in both themes
        palette = {
            "info":    ("#1f3a8a", "#dbeafe"),
            "success": ("#065f46", "#d1fae5"),
            "warning": ("#92400e", "#fef3c7"),
            "error":   ("#7f1d1d", "#fee2e2"),
        }
        fg, bg = palette.get(mode, palette["info"])
        self.banner_var.set(message)
        self.banner_label.configure(foreground=fg, background=bg)
        if message:
            self.banner_frame.grid()
        else:
            self.banner_frame.grid_remove()

    def set_clicks_enabled(self, enabled: bool) -> None:
        self._clicks_enabled = enabled
        self.modified_canvas.configure(cursor="crosshair" if enabled else "arrow")

    def set_reveal_enabled(self, enabled: bool) -> None:
        self.reveal_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_reset_enabled(self, enabled: bool) -> None:
        self.reset_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_hint_enabled(self, enabled: bool) -> None:
        self.hint_button.configure(state=tk.NORMAL if enabled else tk.DISABLED)

    def set_timer_callback(self, callback: Callable[[], None]) -> None:
        self._timer_callback = callback

    def start_timer_loop(self) -> None:
        if self._timer_callback is not None:
            self._timer_callback()
        self.root.after(100, self.start_timer_loop)

    def flash_click(self, x: int, y: int, success: bool) -> None:
        """Briefly show a green check or red cross at the click position."""
        symbol = "✓" if success else "✗"
        colour = "#16a34a" if success else "#dc2626"
        text_id = self.modified_canvas.create_text(
            x, y,
            text=symbol,
            fill=colour,
            font=("TkDefaultFont", 32, "bold"),
            tags=("flash",),
        )
        # Outline-style emphasis via a halo circle
        halo_id = self.modified_canvas.create_oval(
            x - 22, y - 22, x + 22, y + 22,
            outline=colour, width=3,
            tags=("flash",),
        )
        # Fade-out by destroying after a delay (Tkinter has no alpha on items)
        self.root.after(550, lambda: self.modified_canvas.delete(text_id))
        self.root.after(550, lambda: self.modified_canvas.delete(halo_id))

    def show_error(self, title: str, message: str) -> None:
        messagebox.showerror(title, message)

    def show_info(self, title: str, message: str) -> None:
        messagebox.showinfo(title, message)

    def show_round_summary(
        self,
        title: str,
        body: str,
    ) -> None:
        messagebox.showinfo(title, body)

    # ---- magnifier ----------------------------------------------------------

    def update_magnifier(self, zoomed_bgr: np.ndarray | None) -> None:
        """Draw or hide the magnifier lens in the top-right of the modified canvas."""
        self.modified_canvas.delete("magnifier")
        if zoomed_bgr is None:
            return
        self._magnifier_photo = self._to_photo(zoomed_bgr)
        # Place in top-right corner with a small inset
        canvas_w = int(self.modified_canvas["width"])
        inset = 8
        w = self._magnifier_photo.width()
        h = self._magnifier_photo.height()
        self.modified_canvas.create_image(
            canvas_w - inset, inset,
            anchor=tk.NE,
            image=self._magnifier_photo,
            tags=("magnifier",),
        )

    # ---- build / theme ------------------------------------------------------

    def _build_window(self) -> None:
        self.root.title("HIT137 Spot the Difference — Enhanced Edition")
        self.root.geometry("1240x880")
        self.root.minsize(960, 720)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(4, weight=1)

        self._build_header()
        self._build_options_row()
        self._build_counters_row()
        self._build_banner()
        self._build_image_area()
        self._build_footer()

        self._draw_placeholder()

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=(14, 12, 14, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        title_label = ttk.Label(
            header,
            text="Spot the Difference",
            font=("TkDefaultFont", 18, "bold"),
        )
        title_label.grid(row=0, column=0, sticky="w")
        self._title_label = title_label

        controls = ttk.Frame(header)
        controls.grid(row=0, column=1, sticky="e")

        self.load_button = ttk.Button(
            controls, text="Load Image", command=self.controller.load_image_from_dialog
        )
        self.load_button.grid(row=0, column=0, padx=(0, 6))

        self.hint_button = ttk.Button(
            controls, text="Hint (-1 try)", command=self.controller.use_hint
        )
        self.hint_button.grid(row=0, column=1, padx=(0, 6))

        self.reveal_button = ttk.Button(
            controls, text="Reveal", command=self.controller.reveal_differences
        )
        self.reveal_button.grid(row=0, column=2, padx=(0, 6))

        self.reset_button = ttk.Button(
            controls, text="Reset Image", command=self.controller.reset_current_image
        )
        self.reset_button.grid(row=0, column=3)

    def _build_options_row(self) -> None:
        options = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        options.grid(row=1, column=0, sticky="ew")
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Difficulty:").grid(row=0, column=0, sticky="w")
        difficulty_box = ttk.Combobox(
            options,
            textvariable=self.difficulty_var,
            values=("Easy", "Medium", "Hard"),
            state="readonly",
            width=10,
        )
        difficulty_box.grid(row=0, column=1, padx=(6, 14), sticky="w")
        difficulty_box.bind(
            "<<ComboboxSelected>>",
            lambda _e: self.controller.set_difficulty(self.difficulty_var.get()),
        )

        magnifier_check = ttk.Checkbutton(
            options,
            text="Magnifier lens",
            variable=self.magnifier_var,
            command=self._toggle_magnifier,
        )
        magnifier_check.grid(row=0, column=2, padx=(0, 14), sticky="w")

        dark_check = ttk.Checkbutton(
            options,
            text="Dark mode",
            variable=self.dark_mode_var,
            command=self._toggle_dark_mode,
        )
        dark_check.grid(row=0, column=3, sticky="w")

        ttk.Label(options, textvariable=self.timer_var, font=("TkDefaultFont", 11, "bold")).grid(
            row=0, column=4, sticky="e"
        )

    def _build_counters_row(self) -> None:
        counters = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        counters.grid(row=2, column=0, sticky="ew")
        for column in range(6):
            counters.columnconfigure(column, weight=1)

        ttk.Label(counters, textvariable=self.remaining_var).grid(row=0, column=0, sticky="w")
        ttk.Label(counters, textvariable=self.found_var).grid(row=0, column=1, sticky="w")
        ttk.Label(counters, textvariable=self.mistakes_var).grid(row=0, column=2, sticky="w")
        ttk.Label(counters, textvariable=self.hints_var).grid(row=0, column=3, sticky="w")
        ttk.Label(counters, textvariable=self.score_var).grid(row=0, column=4, sticky="e")
        ttk.Label(counters, textvariable=self.best_score_var,
                  font=("TkDefaultFont", 10, "bold")).grid(row=0, column=5, sticky="e")

    def _build_banner(self) -> None:
        self.banner_frame = ttk.Frame(self.root, padding=(14, 0, 14, 8))
        self.banner_frame.grid(row=3, column=0, sticky="ew")
        self.banner_frame.columnconfigure(0, weight=1)
        self.banner_label = tk.Label(
            self.banner_frame,
            textvariable=self.banner_var,
            font=("TkDefaultFont", 11, "bold"),
            padx=12, pady=8, anchor="center",
        )
        self.banner_label.grid(row=0, column=0, sticky="ew")
        self.banner_frame.grid_remove()

    def _build_image_area(self) -> None:
        image_area = ttk.Frame(self.root, padding=(14, 8, 14, 8))
        image_area.grid(row=4, column=0, sticky="nsew")
        image_area.columnconfigure(0, weight=1)
        image_area.columnconfigure(1, weight=1)
        image_area.rowconfigure(1, weight=1)

        ttk.Label(image_area, text="Original").grid(row=0, column=0, pady=(0, 6))
        ttk.Label(image_area, text="Modified (click here)").grid(row=0, column=1, pady=(0, 6))

        self.original_canvas = tk.Canvas(
            image_area, width=520, height=520,
            bg=self._theme["canvas_bg"],
            highlightthickness=1,
            highlightbackground=self._theme["canvas_border"],
        )
        self.original_canvas.grid(row=1, column=0, padx=(0, 8), sticky="n")

        self.modified_canvas = tk.Canvas(
            image_area, width=520, height=520,
            bg=self._theme["canvas_bg"],
            highlightthickness=1,
            highlightbackground=self._theme["canvas_border"],
        )
        self.modified_canvas.grid(row=1, column=1, padx=(8, 0), sticky="n")
        self.modified_canvas.bind("<Button-1>", self._handle_modified_click)
        self.modified_canvas.bind("<Motion>", self._handle_motion)
        self.modified_canvas.bind("<Leave>", self._handle_leave)

    def _build_footer(self) -> None:
        footer = ttk.Frame(self.root, padding=(14, 8, 14, 12))
        footer.grid(row=5, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=0, sticky="w")

    # ---- event handlers -----------------------------------------------------

    def _handle_modified_click(self, event: tk.Event) -> None:
        if not self._clicks_enabled:
            return
        self.controller.handle_modified_click(int(event.x), int(event.y))

    def _handle_motion(self, event: tk.Event) -> None:
        if not self._magnifier_enabled:
            return
        self.controller.handle_magnifier_motion(int(event.x), int(event.y))

    def _handle_leave(self, _event: tk.Event) -> None:
        if self._magnifier_enabled:
            self.update_magnifier(None)

    def _toggle_magnifier(self) -> None:
        self._magnifier_enabled = bool(self.magnifier_var.get())
        if not self._magnifier_enabled:
            self.update_magnifier(None)

    def _toggle_dark_mode(self) -> None:
        self._dark_mode = bool(self.dark_mode_var.get())
        self._theme = DARK_THEME if self._dark_mode else LIGHT_THEME
        self._apply_theme()

    def _apply_theme(self) -> None:
        self.root.configure(bg=self._theme["bg"])
        style = ttk.Style()
        # Try a theme that respects bg colours; fall back gracefully
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=self._theme["bg"])
        style.configure("TLabel", background=self._theme["bg"], foreground=self._theme["fg"])
        style.configure("TButton", padding=6)
        style.configure("TCheckbutton",
                        background=self._theme["bg"], foreground=self._theme["fg"])
        style.configure("TCombobox", fieldbackground=self._theme["canvas_bg"])

        if hasattr(self, "original_canvas"):
            for canvas in (self.original_canvas, self.modified_canvas):
                canvas.configure(
                    bg=self._theme["canvas_bg"],
                    highlightbackground=self._theme["canvas_border"],
                )

    # ---- drawing helpers ----------------------------------------------------

    def _draw_placeholder(self) -> None:
        for canvas in (self.original_canvas, self.modified_canvas):
            canvas.delete("all")
            canvas.configure(width=520, height=520)
            canvas.create_text(
                260, 260,
                text="Load an image",
                fill=self._theme["muted"],
                font=("TkDefaultFont", 14),
            )

    def _draw_photo(self, canvas: tk.Canvas, photo: ImageTk.PhotoImage) -> None:
        canvas.delete("all")
        canvas.configure(width=photo.width(), height=photo.height())
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def _to_photo(self, image_bgr: np.ndarray) -> ImageTk.PhotoImage:
        image_rgb = image_bgr[:, :, ::-1].copy()
        pil_image = Image.fromarray(image_rgb)
        return ImageTk.PhotoImage(pil_image)
