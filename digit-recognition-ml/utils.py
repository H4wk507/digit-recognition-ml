from PIL import Image
import os
import numpy as np


def read_digits(dirname: str):
    """Read digits' images from dirname directory.
    Return a tuple of two lists:
    - X: list of image pixels (each pixel is a number from 0 (white) to 255 (black).
    - y: list of digit labels from 0 to 9.
    Dimensions:
    - X: nsamples * 28 * 28 (each image is resized to 28x28 pixels).
    - y: nsamples."""
    X = []
    y = []
    for subdir, _, files in os.walk(dirname):
        for file in files:
            label = subdir[-1]
            filepath = subdir + os.sep + file
            img = Image.open(filepath)
            img = img.resize((28, 28))
            img = img.convert("L")
            img = np.array(img).flatten()
            img = np.invert(img)
            X.append(img)
            y.append(label)
    return np.array(X), np.array(y)
