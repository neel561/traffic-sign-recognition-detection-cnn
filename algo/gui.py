"""Desktop app for recognising traffic signs, either in a cropped picture or anywhere in a photo."""
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

import detect as detection
import tsr

BACKGROUND = "#2f4f4f"
TITLE_COLOUR = "#f5d020"
RESULT_COLOUR = "#ffc0cb"
BUTTON_COLOUR = "#ffb90f"
FONT = "Georgia"
THUMBNAIL_SIZE = (250, 250)
PHOTO_SIZE = (480, 270)  # annotated photos are shown larger, so the boxes are readable


class TrafficSignApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.detector = None
        self.image = None

        root.title("Road Traffic Sign Recognition")
        # 1200x700 by default, but never bigger than the screen
        root.geometry(f"{min(1200, root.winfo_screenwidth())}x{min(700, root.winfo_screenheight() - 80)}")
        root.minsize(600, 560)
        root.configure(background=BACKGROUND)

        # Packed first so it keeps its space at the bottom however large the selected image is
        self._button("Select a traffic sign", self.select_image).pack(side=tk.BOTTOM, pady=25)
        tk.Label(root, text="Road Traffic Sign Recognition", font=(FONT, 28, "bold"),
                 bg=BACKGROUND, fg=TITLE_COLOUR).pack(pady=20)
        self.result = tk.Label(root, font=(FONT, 20, "bold"), bg=BACKGROUND, fg=RESULT_COLOUR, wraplength=1000)
        self.result.pack(pady=(30, 10))
        self.sign_image = tk.Label(root, bg=BACKGROUND)
        self.sign_image.pack(pady=15)
        self.buttons = tk.Frame(root, bg=BACKGROUND)
        self._button("Recognize the Sign ?", self.recognise, self.buttons).pack(side=tk.LEFT, padx=8)
        self._button("Find signs in photo", self.find_signs, self.buttons).pack(side=tk.LEFT, padx=8)

    def _button(self, text, command, parent=None):
        return tk.Button(parent or self.root, text=text, command=command, font=(FONT, 14, "bold"),
                         bg=BUTTON_COLOUR, fg=BACKGROUND, activebackground=TITLE_COLOUR, padx=18, pady=10)

    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select a traffic sign or a photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.ppm *.webp"), ("All files", "*.*")])
        if path:
            self.show_image(path)

    def _show(self, image, size):
        thumbnail = image.convert("RGBA")
        thumbnail.thumbnail(size)
        photo = ImageTk.PhotoImage(thumbnail)
        self.sign_image.configure(image=photo)
        self.sign_image.image = photo  # keep a reference, otherwise Tk shows a blank image

    def show_image(self, path):
        try:
            image = tsr.open_image(path)
        except (OSError, Image.DecompressionBombError) as error:
            self.image = None
            self.sign_image.configure(image="")
            self.buttons.pack_forget()
            self.result.configure(text="")
            messagebox.showerror("Could not open image", f"{path} could not be read as an image.\n\n{error}")
            return
        self._show(image, THUMBNAIL_SIZE)
        self.image = image
        self.result.configure(text="")
        self.buttons.pack(before=self.sign_image, pady=10)

    def recognise(self):
        """Name the sign, treating the whole picture as one cropped sign."""
        _, sign, confidence = tsr.predict(self.model, self.image)
        if confidence >= tsr.MIN_CONFIDENCE:
            self.result.configure(text=f"{sign}  ({confidence:.0%})")
        else:
            self.result.configure(text=f"No sign recognised  (best guess: {sign}, {confidence:.0%})")

    def find_signs(self):
        """Find every sign in the photo, then name each one."""
        if not detection.DETECTOR_PATH.exists():
            messagebox.showinfo("Detector not trained",
                                "The sign detector has not been trained yet.\n\nRun: python algo/train_detector.py")
            return
        self.result.configure(text="Looking for signs...")
        self.root.update()
        if self.detector is None:
            self.detector = detection.load_detector()
        signs = detection.detect(self.detector, self.model, self.image)
        self._show(detection.annotate(self.image, signs), PHOTO_SIZE)
        named = [s for s in signs if s["recognised"]]
        if not signs:
            self.result.configure(text="No traffic signs found in this photo")
        else:
            found = ", ".join(f"{s['sign']} ({s['confidence']:.0%})" for s in named) or "none could be named"
            missed = len(signs) - len(named)
            self.result.configure(text=f"{len(signs)} sign(s) found: {found}"
                                       + (f", and {missed} not recognised" if named and missed else ""))


def main():
    root = tk.Tk()
    if not tsr.MODEL_PATH.exists():
        root.withdraw()
        messagebox.showerror("Model not found", f"{tsr.MODEL_PATH} is missing.\n\nTrain it first with: python algo/train.py")
        root.destroy()
        return
    TrafficSignApp(root, tsr.load_model())
    root.mainloop()


if __name__ == "__main__":
    main()
