# calibration_gui.py
import tkinter as tk
import time

class CalibrationGUI:
    """
    Fullscreen Tkinter calibration UI.
    Calls `on_point_callback(name, nx, ny)` synchronously for each calibration point.
    The callback is expected to collect samples for the duration set here (default 2s).
    """

    DEFAULT_POINTS = [
        ("CENTER", 0.50, 0.50),
        ("LEFT",   0.15, 0.50),
        ("RIGHT",  0.85, 0.50),
        ("TOP",    0.50, 0.20),
        ("BOTTOM", 0.50, 0.80),
    ]

    def __init__(self, on_point_callback, duration_per_point=2.0, points=None):
        """
        on_point_callback(name, nx, ny) -> should collect samples for duration_per_point seconds
        duration_per_point: seconds to display each point
        points: list of (name, nx, ny) normalized 0..1
        """
        self.on_point_callback = on_point_callback
        self.duration = float(duration_per_point)
        self.CALIB_POINTS = points if points is not None else self.DEFAULT_POINTS

        self.root = tk.Tk()
        self.root.title("Calibration")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")

        # detect screen size
        self.SCREEN_W = self.root.winfo_screenwidth()
        self.SCREEN_H = self.root.winfo_screenheight()

        # Start button
        self.start_btn = tk.Button(
            self.root,
            text="START CALIBRATION",
            font=("Arial", 44, "bold"),
            bg="white",
            fg="black",
            padx=40,
            pady=20,
            command=self._on_start_clicked
        )
        self.start_btn.pack(expand=True)

        # Optional abort keybinding (Esc)
        self.root.bind("<Escape>", lambda e: self.root.destroy())

    def _draw_point(self, name, nx, ny):
        """Show a calibration point fullscreen. Returns the canvas (so the caller can update if needed)."""
        for w in self.root.winfo_children():
            w.destroy()

        canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        x = int(nx * self.SCREEN_W)
        y = int(ny * self.SCREEN_H)
        r = 40
        canvas.create_oval(x - r, y - r, x + r, y + r, fill="red", outline="")

        canvas.create_text(
            self.SCREEN_W//2, int(self.SCREEN_H*0.12),
            text=f"Look at {name}",
            fill="white",
            font=("Arial", 36, "bold")
        )

        # small instruction
        canvas.create_text(
            self.SCREEN_W//2, int(self.SCREEN_H*0.9),
            text=f"Collecting samples for {self.duration:.1f}s — don't move your head much",
            fill="gray",
            font=("Arial", 22)
        )

        self.root.update()
        return canvas

    def _on_start_clicked(self):
        # remove button and start sequence
        self.start_btn.destroy()
        # run each point synchronously: draw -> call callback -> wait duration
        for name, nx, ny in self.CALIB_POINTS:
            self._draw_point(name, nx, ny)
            # callback should itself collect samples for the duration (synchronous)
            try:
                self.on_point_callback(name, nx, ny)
            except Exception as e:
                # If callback raises, print (callback runs in main process)
                print("[CalibrationGUI] Warning: on_point_callback raised:", e)
            # brief small pause to let GUI update text (callback likely consumed time)
            time.sleep(0.05)

        # close GUI after calibration ends
        self.root.destroy()

    def start(self):
        self.root.mainloop()
