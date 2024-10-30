import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the binary image
def load_image(file_path, width, height):
    with open(file_path, 'rb') as f:
        img = np.fromfile(f, dtype=np.uint8, count=width * height)
    return img.reshape((height, width))

# Step 2: Plot the histogram of the image
def plot_histogram(image, ax, title):
    ax.hist(image.flatten(), bins=256, range=[0, 256], color='blue')
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Frequency')
    ax.grid()

# Step 3: Perform histogram equalization
def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized).reshape(image.shape)
    return image_equalized.astype(np.uint8)

# Main program
width, height = 256, 256
image_path = 'johnnybin.sec'

# Load the original image
original_image = load_image(image_path, width, height)

# Perform histogram equalization
equalized_image = histogram_equalization(original_image)

# Create a figure for displaying images and histograms with larger size
fig, axs = plt.subplots(2, 3, figsize=(18, 12))  # Increased figure size

# Show original image
axs[0, 0].imshow(original_image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Show equalized image
axs[0, 1].imshow(equalized_image, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[0, 1].axis('off')

# Plot histogram of the original image
plot_histogram(original_image, axs[1, 0], 'Histogram of Original Image')

# Plot histogram of the equalized image
plot_histogram(equalized_image, axs[1, 1], 'Histogram of Equalized Image')

# Hide empty subplots
axs[0, 2].axis('off')
axs[1, 2].axis('off')

# Center the images and histograms
for ax in axs.flat:
    ax.set_box_aspect(1)  # Set aspect ratio to be equal

# Adjust layout to center everything and show the plot
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)
plt.show()
