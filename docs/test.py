from PIL import Image

with Image.open("number_example.png") as im:
    im.resize((28, 28)).convert("L").save("normalized.png", compress_level=0)
