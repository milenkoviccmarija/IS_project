from PIL import Image
import matplotlib.pyplot as plt

images = [
    "runs/marija_masa_model/results.png",
    "runs/marija_masa_model/confusion_matrix.png",
    "runs/marija_masa_model/F1_curve.png",
    "runs/marija_masa_model/PR_curve.png",
]

for path in images:
    img = Image.open(path)

    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis("off")
    plt.title(path.split("/")[-1])

    plt.show()