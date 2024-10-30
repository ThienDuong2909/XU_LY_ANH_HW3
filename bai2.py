import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
with open("ladybin.sec", "rb") as f:
    image = np.fromfile(f, dtype=np.uint8).reshape((256, 256))

# Perform full-scale contrast stretch
min_pixel = image.min()
max_pixel = image.max()
stretched_image = ((image - min_pixel) * (255 / (max_pixel - min_pixel))).astype(np.uint8)

# Plot both images and their histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Display the original image
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

# Histogram of the original image
axes[1, 0].hist(image.ravel(), bins=256, range=(0, 255), color='blue')
axes[1, 0].set_title("Histogram of Original Image")
axes[1, 0].set_xlabel("Pixel Intensity")
axes[1, 0].set_ylabel("Frequency")

# Display the contrast-stretched image
axes[0, 1].imshow(stretched_image, cmap='gray')
axes[0, 1].set_title("Contrast-Stretched Image")
axes[0, 1].axis("off")

# Histogram of the contrast-stretched image
axes[1, 1].hist(stretched_image.ravel(), bins=256, range=(0, 255), color='blue')
axes[1, 1].set_title("Histogram of Contrast-Stretched Image")
axes[1, 1].set_xlabel("Pixel Intensity")
axes[1, 1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
