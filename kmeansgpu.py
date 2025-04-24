import cupy as cp
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import time

# Load the image with transparency (RGBA)
img = Image.open('img/crossline.jpg').convert('RGBA')
inp = np.array(img)  # Pixel values in RGBA format
alpha_channel = inp[:, :, 3]  # Extract the alpha channel
rgb_values = inp[:, :, :3]  # Extract the RGB values

# Flatten the image for clustering
flat_rgb = rgb_values.reshape(-1, 3)  # Shape: (num_pixels, 3)
flat_alpha = alpha_channel.flatten()  # Shape: (num_pixels,)

# Mask out fully transparent pixels
non_transparent_mask = flat_alpha > 0  # True for non-transparent pixels
filtered_rgb = flat_rgb[non_transparent_mask]  # Only non-transparent pixels

# Convert to CuPy array for clustering
Y = cp.array(filtered_rgb.T)  # Shape: (3, num_non_transparent_pixels)
m, N = Y.shape  # Number of parameters and pixels
k = 4  # Number of clusters for the first k-means
k2 = 3  # Number of clusters for the second k-means
epsilon = 0.5  # Stopping condition
batch_size = 10000  # Batch size for distance computation

# Image dimensions
original_width, original_height = img.size

# Start time measurement
start_time = time.time()

# Do iterations to find the centroids of the clusters
def color_Iteration(init_centroids, k):
    ite = 1
    while ite < k:
        max_sq_dist = -1
        farthest_point_index = -1
        distances = cp.min(cp.sum((Y[:, :, None] - init_centroids[:, None, :ite]) ** 2, axis=0), axis=1)
        farthest_point_index = cp.argmax(distances)
        max_sq_dist = distances[farthest_point_index]
        init_centroids[:, ite] = Y[:, farthest_point_index]
        ite += 1
    return init_centroids.copy()

# Function to perform k-means clustering
def error_iteration(centroids, k, batch_size=10000):
    error = 10000  # Initial error
    q = 0  # Iteration counter
    while error > epsilon:
        cluster_assignments = cp.zeros(N, dtype=cp.int32)
        distances = cp.zeros((k, N), dtype=cp.float32)

        # Compute distances in batches
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_Y = Y[:, start:end]  # Process a batch of pixels
            # Correct computation of batch distances
            batch_distances = cp.sqrt(cp.sum((batch_Y[:, :, None] - centroids[:, None, :]) ** 2, axis=0))
            distances[:, start:end] = batch_distances.T  # Transpose to match the expected shape

        cluster_assignments = cp.argmin(distances, axis=0)
        new_centroids = cp.zeros_like(centroids)

        for i in range(k):
            mask = cluster_assignments == i
            points_in_cluster = Y[:, mask]
            if points_in_cluster.size > 0:
                new_centroids[:, i] = cp.mean(points_in_cluster, axis=1)

        error = cp.sqrt(cp.sum((centroids - new_centroids) ** 2))
        centroids = new_centroids
        q += 1
        print(f"Iteration {q}: Error = {error}")
    return centroids, cluster_assignments

# First K-Means
print("Running first k-means...")

init_centroids = cp.zeros((3, k))
init_centroids[:, 0] = Y[:, cp.random.randint(0, N)]  # First centroid is random

centroids = color_Iteration(init_centroids, k)  # Initialize centroids using farthest point method

# Perform k-means clustering
centroids, cluster_assignments = error_iteration(centroids, k)

# End time measurement
end_time = time.time()
print(f"First k-means completed in {end_time - start_time:.2f} seconds.")

# Reconstruct the full image with clustered colors
clustered_flat_rgb = cp.asnumpy(centroids[:, cluster_assignments].T.astype(np.uint8))  # Convert back to NumPy
reconstructed_rgb = np.zeros((flat_rgb.shape[0], 3), dtype=np.uint8)  # Initialize with zeros
reconstructed_rgb[non_transparent_mask] = clustered_flat_rgb  # Map clustered colors to non-transparent pixels

# Combine with the original alpha channel
reconstructed_image = np.zeros((flat_alpha.size, 4), dtype=np.uint8)  # RGBA
reconstructed_image[:, :3] = reconstructed_rgb  # Add RGB values
reconstructed_image[:, 3] = flat_alpha  # Add the original alpha channel

# Reshape back to the original image dimensions
reconstructed_image = reconstructed_image.reshape(original_height, original_width, 4)

# Save the reconstructed image
Image.fromarray(reconstructed_image, 'RGBA').save('clustered_image_with_alpha.png')


# Identify the largest clusters (e.g., black and yellow)
cluster_sizes = [cp.sum(cluster_assignments == i).get() for i in range(k)]  # Convert to NumPy arrays
largest_clusters = np.argsort(cluster_sizes)[-3:]  # Indices of the two largest clusters
print(f"Largest clusters: {largest_clusters}")


print(f"Unique cluster assignments: {np.unique(cluster_assignments.get())}")
print(f"Shape of non_transparent_mask: {non_transparent_mask.shape}")
print(f"Number of non-transparent pixels: {np.sum(non_transparent_mask)}")



# Visualize each cluster separately from the first k-means
if 1:  # Set to True to visualize clusters
    # Convert clustered pixels back to NumPy
    clustered_flat_rgb = cp.asnumpy(centroids[:, cluster_assignments].T.astype(np.uint8))

    # Reconstruct the full image with clustered colors
    reconstructed_rgb = np.zeros((flat_rgb.shape[0], 3), dtype=np.uint8)  # Initialize with zeros
    reconstructed_rgb[non_transparent_mask] = clustered_flat_rgb  # Map clustered colors to non-transparent pixels

    # Reshape back to the original image dimensions
    clustered_image = reconstructed_rgb.reshape(original_height, original_width, 3)

    # Save the clustered image
    cv2.imwrite("clustered_image.png", cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))  # Save the clustered image
    plt.imshow(clustered_image)
    plt.axis('off')
    plt.title("First K-Means Result")
    plt.show()  # Add this line to display the plot
for i in range(k):
    # Create a blank RGBA image for the full dataset
    cluster_image_flat = np.zeros((flat_alpha.size, 4), dtype=np.uint8)  # Initialize with zeros

    # Get the mask for the current cluster
    cluster_mask = (cluster_assignments == i)  # Boolean mask for the current cluster (non-transparent pixels only)

    # Create a combined mask
    combined_mask = np.zeros(flat_alpha.size, dtype=bool)  # Initialize with all False
    combined_mask[non_transparent_mask] = cp.asnumpy(cluster_mask)  # Set True only for non-transparent pixels in the current cluster

    # Get the centroid color for the current cluster
    centroid_color = cp.asnumpy(centroids[:, i].T)

    # Assign the centroid color to the pixels in the current cluster
    cluster_image_flat[combined_mask, :3] = centroid_color  # Assign RGB values

    # Set alpha to 255 for pixels in the current cluster, 0 for others
    cluster_image_flat[combined_mask, 3] = 255  # Set alpha to 255 for the current cluster
    cluster_image_flat[~combined_mask, 3] = 0  # Set alpha to 0 for other pixels

    # Reconstruct the full image with the cluster's colors
    cluster_image = cluster_image_flat.reshape(original_height, original_width, 4)

    # Save or display the cluster image
    cluster_filename = f"cluster_{i}.png"

    # Save the image
    Image.fromarray(cluster_image, 'RGBA').save(cluster_filename)
    print(f"Saved cluster {i} as {cluster_filename}")

    # Optionally display the cluster image
    plt.imshow(cluster_image)
    plt.axis('off')
    plt.title(f"Cluster {i}")
    plt.show()

# Mask out the largest clusters in the original dataset
print("Masking out the largest clusters...")
mask = cp.ones(N, dtype=bool)
for cluster in largest_clusters:
    mask &= cluster_assignments != cluster  # Exclude the largest clusters
masked_Y = Y[:, mask]  # Remaining pixels for the second k-means

# Save the original image with the largest clusters masked out
print("Saving the original image with the largest clusters masked out...")

# Initialize a blank image for the full dataset
masked_image_np = np.zeros((flat_rgb.shape[0], 3), dtype=np.uint8)  # Initialize with zeros

# Create a new mask for the remaining pixels
remaining_mask = np.zeros(flat_rgb.shape[0], dtype=bool)  # Initialize with all False
remaining_mask[non_transparent_mask] = mask.get()  # Update with the mask for remaining pixels

# Map the remaining pixels back to their original positions
remaining_pixels = cp.asnumpy(Y[:, mask].T.astype(np.uint8))  # Convert remaining pixels to NumPy
masked_image_np[remaining_mask] = remaining_pixels  # Map remaining pixels to their correct positions

# Reshape back to the original image dimensions
masked_image_np = masked_image_np.reshape(original_height, original_width, 3)

# Save the masked image
cv2.imwrite("masked_image.png", cv2.cvtColor(masked_image_np, cv2.COLOR_RGB2BGR))  # Save the masked image

# Second K-Means
print("Running second k-means on the original image with largest clusters masked out...")

init_centroids_2 = cp.zeros((3, k2))
init_centroids_2[:, 0] = masked_Y[:, cp.random.randint(0, masked_Y.shape[1])]

centroids_2 = color_Iteration(init_centroids_2, k2)  # Initialize centroids using farthest point method

centroids_2, cluster_assignments_2 = error_iteration(centroids_2, k2)  # Perform k-means clustering

# Convert clustered image back to NumPy for visualization
clustered_image_2 = cp.asnumpy(centroids[:, cluster_assignments_2].T.reshape(img.size[1], img.size[0], 3).astype(np.uint8))
cv2.imwrite("clustered_image_2.png", cv2.cvtColor(clustered_image_2, cv2.COLOR_RGB2BGR))  # Save the clustered image
plt.imshow(clustered_image_2)
plt.axis('off')
plt.title("Second K-Means Result")
plt.show()  # Add this line to display the plot

# Map the second k-means results back to the original image
print("Mapping second k-means results back to the original image...")
clustered_image_2 = np.zeros((original_height, original_width, 3), dtype=np.uint8)
cluster_assignments_2_np = cluster_assignments_2.get()  # Convert to NumPy
centroids_2_np = cp.asnumpy(centroids_2)  # Convert centroids to NumPy

# Debugging: Print cluster centroids
print(f"Cluster centroids (second k-means): {centroids_2_np}")

# Assign the cluster colors to the filtered pixels
filtered_indices = np.where(mask.get())[0]  # Indices of remaining pixels in the original dataset
clustered_image_2_flat = np.zeros((N, 3), dtype=np.uint8)

for i in range(k2):
    # Create a mask for the current cluster within the filtered dataset
    cluster_mask = (cluster_assignments_2_np == i)  # Boolean mask for the current cluster

    # Get the indices of the pixels in the current cluster within the filtered dataset
    cluster_filtered_indices = np.where(cluster_mask)[0]  # Indices in the filtered dataset

    if len(cluster_filtered_indices) > 0:
        # Map these indices back to the original dataset
        cluster_indices = filtered_indices[cluster_filtered_indices]

        # Debugging: Check cluster_indices
        print(f"Cluster {i}: Mapped indices in original dataset: {cluster_indices[:10]} (showing first 10)")

        # Assign the cluster's centroid color to the corresponding pixels
        clustered_image_2_flat[cluster_indices] = centroids_2_np[:, i].T

# Debugging: Check unique values in clustered_image_2_flat
print(f"Unique values in clustered_image_2_flat: {np.unique(clustered_image_2_flat, axis=0)}")

# Reshape the flat image back to the original dimensions
clustered_image_2 = clustered_image_2_flat.reshape(original_height, original_width, 3)

# Debugging: Check reshaped image dimensions
print(f"Shape of clustered_image_2_flat: {clustered_image_2_flat.shape}")
print(f"Expected shape after reshaping: ({original_height}, {original_width}, 3)")

# Display the second k-means result
plt.imshow(clustered_image_2)
plt.axis('off')
plt.title("Second K-Means Result (Largest Clusters Masked Out)")
plt.show()

# Reconstruct the full image with clustered colors
clustered_flat_rgb = cp.asnumpy(centroids[:, cluster_assignments].T.astype(np.uint8))  # Convert back to NumPy
reconstructed_rgb = np.zeros((flat_rgb.shape[0], 3), dtype=np.uint8)  # Initialize with zeros
reconstructed_rgb[non_transparent_mask] = clustered_flat_rgb  # Map clustered colors to non-transparent pixels

# Combine with the original alpha channel
reconstructed_image = np.zeros((flat_alpha.size, 4), dtype=np.uint8)  # RGBA
reconstructed_image[:, :3] = reconstructed_rgb  # Add RGB values
reconstructed_image[:, 3] = flat_alpha  # Add the original alpha channel

# Reshape back to the original image dimensions
reconstructed_image = reconstructed_image.reshape(original_height, original_width, 4)

# Save the reconstructed image
Image.fromarray(reconstructed_image, 'RGBA').save('clustered_image_with_alpha.png')