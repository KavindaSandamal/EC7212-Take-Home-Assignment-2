# Sandamal IMK - EG/2020/4190
# --  Otsu's Thresholding on a Synthetic Image with Gaussian Noise --
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, sigma=20):
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = image + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

# Create blank image
height, width = 200, 200
image = np.zeros((height, width), dtype=np.uint8)  # Background = 0
obj1_val = 90    # Rectangle 1
obj2_val = 160   # Rectangle 2
cv2.rectangle(image, (40, 50), (100, 110), obj1_val, -1)    # Object 1
cv2.rectangle(image, (120, 130), (180, 180), obj2_val, -1)  # Object 2

# Add Gaussian noise
noisy_image = add_gaussian_noise(image)

# Apply Otsu’s thresholding
_, otsu_thresh = cv2.threshold(noisy_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save images
output_dir = 'Outputs'

os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(os.path.join(output_dir, 'image_original.png'), image)
cv2.imwrite(os.path.join(output_dir, 'image_noisy.png'), noisy_image)
cv2.imwrite(os.path.join(output_dir, 'image_otsu.png'), otsu_thresh)

# Plot images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Otsu's Threshold")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

# Save combined plot
combined_path = os.path.join(output_dir, 'combined_result.png')
plt.tight_layout()
plt.savefig(combined_path)
print(f"Combined plot saved at: {combined_path}")

# Show plot
plt.show()
