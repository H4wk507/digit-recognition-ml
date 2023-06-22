import os
import tkinter as tk
from datetime import datetime

import numpy as np
from PIL import Image, ImageGrab  # Pillow
from sklearn.neighbors import KNeighborsClassifier  # scikit-learn
from utils import read_digits


class Settings:
    WIDTH = HEIGHT = 300
    FONTSIZE = 20


def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert("L")
    img = np.array(img).flatten()
    img = np.invert(img)
    img = img / 255
    x_train, y_train = read_digits("imgs")
    x_train = x_train / 255
    model = KNeighborsClassifier(n_neighbors=3, p=1)
    model.fit(x_train, y_train)
    y_pred = model.predict([img])[0]
    return y_pred


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        self.canvas = tk.Canvas(
            self,
            width=Settings.WIDTH,
            height=Settings.HEIGHT,
            bg="white",
            cursor="cross",
        )
        self.label = tk.Label(
            self, text="?", font=("Helvetica", Settings.FONTSIZE)
        )
        self.classify_button = tk.Button(
            self, text="Recognize", command=self.classify_handwriting
        )
        self.clear_button = tk.Button(
            self, text="Clear", command=self.clear_all
        )
        self.save_button = tk.Button(
            self, text="Save", command=self.save_to_file
        )
        self.canvas.grid(
            row=0,
            column=0,
            pady=2,
            sticky=tk.W,
        )
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_button.grid(row=1, column=1, pady=2, padx=2)
        self.clear_button.grid(row=1, column=0, pady=2)
        self.save_button.grid(row=1, column=2, pady=2)
        self.canvas.bind("<B1-Motion>", self.draw_lines)

    def clear_all(self):
        self.canvas.delete("all")

    def get_canvas_image(self) -> Image:
        x = self.winfo_rootx() + self.canvas.winfo_x()
        y = self.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        return ImageGrab.grab().crop((x, y, x1, y1))

    def classify_handwriting(self):
        im = self.get_canvas_image()
        digit = predict_digit(im)
        self.label.configure(text=str(digit))

    def save_to_file(self):
        folder = "imgs"
        filename = datetime.today().strftime("%d-%m-%Y %Hh%Mm%Ss") + ".png"
        im = self.get_canvas_image()
        im.save(os.path.join(folder, filename))

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 20
        self.canvas.create_oval(
            self.x - r,
            self.y - r,
            self.x + r,
            self.y + r,
            fill="black",
            outline="black",
        )


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
