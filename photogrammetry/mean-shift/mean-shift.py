import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_mean_shift(image, target_size=(512, 512), spatial_radius=10, color_radius=10, max_level=1):
    # Resize image to a standard size
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Apply mean shift filtering
    mean_shifted = cv2.pyrMeanShiftFiltering(image_resized, spatial_radius, color_radius, max_level)

    return image_resized, mean_shifted

# Load image
image_path = r"C:\Users\HDSL77\Desktop\DSC_2541.jpg"
image = cv2.imread(image_path)

# Compute mean shift
resized_image, mean_shifted_image = compute_mean_shift(image)

# Show original and filtered images
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Resized Original")
plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Mean-Shifted")
plt.imshow(cv2.cvtColor(mean_shifted_image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
