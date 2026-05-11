"""main.py

Owner: Avery + Jane
Responsibility: application entry point and startup flow.
Notes: This file will later create and launch the main game controller.
"""

from __future__ import annotations

import sys
import tkinter as tk
from tkinter import messagebox


def main() -> int:
    root = tk.Tk()

    try:
        from game_controller import GameController
    except ImportError as exc:
        messagebox.showerror(
            "Missing dependency",
            "The game requires OpenCV, NumPy, and Pillow.\n\n"
            "Install them with:\n"
            "python3 -m pip install -r requirements.txt\n\n"
            f"Details: {exc}",
        )
        root.destroy()
        return 1

    app = GameController(root)
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
