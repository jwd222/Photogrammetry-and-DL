import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def mean_shift_vectorized_gpu(image, spatial_radius=5, color_radius=20, max_iter=5, target_size=(128, 128)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resize and normalize
    image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(image_resized).float().to(device) / 255.0  # [H, W, C]
    H, W, C = img.shape
    flat_img = img.permute(2, 0, 1).reshape(C, -1).T  # [N, 3]

    # Create pixel grid (spatial positions)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([yy, xx], dim=2).reshape(-1, 2).float().to(device)  # [N, 2]

    # Combine spatial and color into one feature vector
    color_bw = color_radius / 255.0
    features = torch.cat([flat_img / color_bw, grid / spatial_radius], dim=1)  # [N, 5]

    shifted = features.clone()

    for _ in range(max_iter):
        dist = torch.cdist(shifted, features)  # [N, N] pairwise distances
        mask = dist < 1.0  # [N, N] neighborhood mask

        numerators = torch.matmul(mask.float(), features)  # weighted sum of neighbors
        counts = mask.sum(dim=1, keepdim=True)
        counts[counts == 0] = 1  # prevent division by zero
        shifted = numerators / counts  # mean of neighbors

    # Extract final colors and reshape to image
    result_color = (shifted[:, :3] * color_bw).clamp(0, 1)
    result_img = result_color.reshape(H, W, C).cpu().numpy()
    result_img = (result_img * 255).astype(np.uint8)

    return image_resized, result_img

# Load image
image_path = r"C:\Users\HDSL77\Desktop\DSC_2541.jpg"
image = cv2.imread(image_path)

# Apply vectorized mean shift
resized_img, mean_shifted_img = mean_shift_vectorized_gpu(image, spatial_radius=100, color_radius=50, max_iter=5, target_size=(160, 160))

# Plot
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Resized Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Mean Shifted (GPU Vectorized)")
plt.imshow(cv2.cvtColor(mean_shifted_img, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.tight_layout()
plt.show()
