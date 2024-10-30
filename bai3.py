import numpy as np
import cv2
import matplotlib.pyplot as plt

# Bước 1: Đọc ảnh nhị phân
img = np.fromfile('actontBinbin.sec', dtype=np.uint8).reshape((256, 256))

# Bước 2: Thiết kế mẫu chữ "T"
# Mẫu chữ "T" có thể được thiết kế theo kích thước 5x5
template = np.array([[0, 0, 255, 0, 0],
                     [0, 0, 255, 0, 0],
                     [255, 255, 255, 255, 255],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]], dtype=np.uint8)

# Kích thước của mẫu
template_height, template_width = template.shape

# Bước 3: Tính toán độ đo khớp M2
M2 = np.zeros_like(img, dtype=np.float32)

for i in range(template_height // 2, img.shape[0] - template_height // 2):
    for j in range(template_width // 2, img.shape[1] - template_width // 2):
        # Lấy vùng lân cận
        region = img[i - template_height // 2:i + template_height // 2 + 1,
                 j - template_width // 2:j + template_width // 2 + 1]

        # Tính toán độ đo khớp M2
        match_value = np.sum(region == template)
        M2[i, j] = match_value

# Bước 4: Tạo ảnh đầu ra J1
J1 = np.zeros_like(M2)
J1[M2 > 0] = M2[M2 > 0]  # Giữ lại giá trị M2 cho những pixel có vùng lân cận đủ lớn

# Bước 5: Ngưỡng hóa ảnh J1
threshold = 10  # Ngưỡng tùy chọn
J2 = np.where(J1 > threshold, 255, 0).astype(np.uint8)

# Hiển thị kết quả
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')

plt.subplot(1, 3, 2)
plt.title('Match Measure J1')
plt.imshow(J1, cmap='gray')

plt.subplot(1, 3, 3)
plt.title('Thresholded Image J2')
plt.imshow(J2, cmap='gray')

plt.show()
