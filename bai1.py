import numpy as np
import matplotlib.pyplot as plt

# Load the image
with open("mammogrambin.sec", "rb") as f:
    original_image = np.fromfile(f, dtype=np.uint8).reshape((512, 512))

# Step (a): Convert the original image to a binary image using a threshold
threshold = 128  # Adjust this value if needed
binary_image = np.where(original_image > threshold, 255, 0).astype(np.uint8)

# Step (b): Generate the contour image from the binary image
contour_image = np.zeros_like(binary_image)
for i in range(1, binary_image.shape[0] - 1):
    for j in range(1, binary_image.shape[1] - 1):
        if binary_image[i, j] == 255:
            # Check if any neighbor is 0 (indicating an edge)
            if (binary_image[i-1:i+2, j-1:j+2] == 0).any():
                contour_image[i, j] = 255

# Display all three images in a single log
plt.figure(figsize=(15, 5))

# Original image
plt.subplot(1, 3, 1)
plt.imshow(original_image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

# Binary image
plt.subplot(1, 3, 2)
plt.imshow(binary_image, cmap='gray')
plt.title("Binary Image")
plt.axis("off")

# Contour image
plt.subplot(1, 3, 3)
plt.imshow(contour_image, cmap='gray')
plt.title("Contour Image")
plt.axis("off")

plt.show()

print("Câu hỏi: Chain code có thể được sử dụng để đại diện cho đường viền chính trong ảnh không?")
print(
    "Trả lời: Có, mã chuỗi có thể được sử dụng để đại diện cho đường viền chính vì nó là phương pháp mô tả đường biên của đối tượng. "
    "Tuy nhiên, nếu đường viền có nhiễu hoặc các khoảng trống nhỏ, mã chuỗi sẽ trở nên phức tạp và có thể yêu cầu làm mịn để đại diện chính xác.")