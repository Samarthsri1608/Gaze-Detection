# calibration_gui.py
import tkinter as tk
import time

class CalibrationGUI:
    def __init__(self, on_point_callback, duration_per_point=2):
        """
        on_point_callback(name, nx, ny)
        name: point label
        nx, ny: normalized coordinates (0–1)
        """
        self.on_point_callback = on_point_callback
        self.duration = duration_per_point

        # Calibration points (normalized positions)
        self.CALIB_POINTS = [
            ("CENTER", 0.50, 0.50),
            ("LEFT",   0.15, 0.50),
            ("RIGHT",  0.85, 0.50),
            ("TOP",    0.50, 0.20),
            ("BOTTOM", 0.50, 0.80),
        ]

        # Create fullscreen window
        self.root = tk.Tk()
        self.root.title("Calibration")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="black")

        # Detect screen size
        self.SCREEN_W = self.root.winfo_screenwidth()
        self.SCREEN_H = self.root.winfo_screenheight()

        # Start button
        self.start_btn = tk.Button(
            self.root,
            text="START CALIBRATION",
            font=("Arial", 40, "bold"),
            bg="white",
            fg="black",
            command=self.start_calibration
        )
        self.start_btn.pack(expand=True)

    def start_calibration(self):
        self.start_btn.destroy()
        self.run_points()

    def show_point(self, name, nx, ny):
        """Draw single calibration dot and text."""
        for widget in self.root.winfo_children():
            widget.destroy()

        canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        x = int(nx * self.SCREEN_W)
        y = int(ny * self.SCREEN_H)
        r = 35  # dot radius

        canvas.create_oval(x-r, y-r, x+r, y+r, fill="red")
        canvas.create_text(
            self.SCREEN_W//2, self.SCREEN_H//8,
            text=f"Look at {name}",
            fill="white",
            font=("Arial", 38, "bold")
        )

        self.root.update()

    def run_points(self):
        for name, nx, ny in self.CALIB_POINTS:
            # Show dot
            self.show_point(name, nx, ny)

            # Tell proctor script to collect samples
            self.on_point_callback(name, nx, ny)

            # Keep the dot for duration
            time.sleep(self.duration)

        # Close GUI after calibration
        self.root.destroy()

    def start(self):
        self.root.mainloop()
