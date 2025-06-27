# Sandamal IMK - EG/2020/4190
# -- Region Growing-Based Image Segmentation --

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import os

# Region Growing Function
def region_growing(image, seeds, threshold=20):
    h, w = image.shape
    segmented = np.zeros_like(image, dtype=np.uint8)
    visited = np.zeros_like(image, dtype=bool)
    queue = deque(seeds)

    while queue:
        y, x = queue.popleft()
        if visited[y, x]:
            continue

        visited[y, x] = True
        segmented[y, x] = 255

        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-neighborhood
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                if abs(int(image[ny, nx]) - int(image[y, x])) <= threshold:
                    queue.append((ny, nx))

    return segmented

# Load a real grayscale image
image_path = 'D:\EC7212_Assignment_2_4190\Image\original_image.jpg'  
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Define seed points
h, w = image.shape
seed_y, seed_x = h // 2, w // 2  # Center seed 
seeds = [(seed_y, seed_x)]

# Run region growing segmentation
segmented_image = region_growing(image, seeds, threshold=15)

# Save and display results
output_dir = 'Outputs'
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(os.path.join(output_dir, 'real_image_grayscale.png'), image)
cv2.imwrite(os.path.join(output_dir, 'region_growing_result.png'), segmented_image)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Region Growing Segmentation')
plt.imshow(segmented_image, cmap='gray')
plt.axis('off')

combined_path = os.path.join(output_dir, 'region_growing_result.png')
plt.tight_layout()
plt.savefig(combined_path)
print(f'Combined plot saved to: {combined_path}')

plt.show()
