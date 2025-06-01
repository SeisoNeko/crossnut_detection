import cv2
import time
import os
from cv2.typing import MatLike
from PIL import Image
from matplotlib import pyplot as plt
from pathlib import Path

import numpy as np
cp = None
yellow_lower = np.array([20, 80, 50], np.uint8)
yellow_upper = np.array([35, 255, 255], np.uint8)


# Do iterations to find the centroids of the clusters
def color_Iteration(init_centroids, k, Y):
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
def error_iteration(centroids, k, N, Y, batch_size=10000, epsilon=0.5):
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
        # print(f"Iteration {q}: Error = {error}")
    return centroids, cluster_assignments


def kmeans_gpu(img: MatLike, device: str) -> MatLike:
    global cp
    if device == "cuda": import cupy as cp
    else: cp = np
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    # cover the yellow area with black and set to transparent
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # Convert to BGRA format for transparency
    img[yellow_mask > 0] = (0, 0, 0, 0)  # Set yellow pixels to transparent  
    
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)  # Convert to RGB format for processing
    
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
    k = 1  # Number of clusters for the first k-means


    # Image dimensions
    original_height, original_width = img.shape[:2]  # Get original dimensions

    # Start time measurement
    start_time = time.time()

    # K-Means
    # print("Running k-means...")

    init_centroids = cp.zeros((3, k))
    init_centroids[:, 0] = Y[:, cp.random.randint(0, N)]  # First centroid is random

    centroids = color_Iteration(init_centroids, k, Y)  # Initialize centroids using farthest point method

    # Perform k-means clustering
    centroids, cluster_assignments = error_iteration(centroids, k, N, Y)

    # End time measurement
    end_time = time.time()
    # print(f"k-means completed in {end_time - start_time:.2f} seconds.")

    # Reconstruct the full image with clustered colors
    clustered_flat_rgb = None
    if (cp == np):
        clustered_flat_rgb = centroids[:, cluster_assignments].T.astype(np.uint8)
    else:
        clustered_flat_rgb = cp.asnumpy(centroids[:, cluster_assignments].T.astype(np.uint8))  # Convert back to NumPy
    
    reconstructed_rgb = np.zeros((flat_rgb.shape[0], 3), dtype=np.uint8)  # Initialize with zeros
    reconstructed_rgb[non_transparent_mask] = clustered_flat_rgb  # Map clustered colors to non-transparent pixels

    # Combine with the original alpha channel
    reconstructed_image = np.zeros((flat_alpha.size, 4), dtype=np.uint8)  # RGBA
    reconstructed_image[:, :3] = reconstructed_rgb  # Add RGB values
    reconstructed_image[:, 3] = flat_alpha  # Add the original alpha channel

    # Reshape back to the original image dimensions
    reconstructed_image = reconstructed_image.reshape(original_height, original_width, 4)

    # Return the reconstructed image    
    # Image.fromarray(reconstructed_image, 'RGBA').save('clustered_image_with_alpha.png')
    return cv2.cvtColor(reconstructed_image, cv2.COLOR_RGBA2BGRA)
    
    # Visualize each cluster separately from the first k-means
    if False:  # Set to True to visualize clusters
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
            # Save the reconstructed image

    
if __name__ == '__main__':
    import os
    
    input_dir = './temp/labels/PXL_20250415_093702508/'  # Directory containing cropped images

    for image_file in os.listdir(input_dir):
        if not image_file.endswith('.png') or image_file.endswith('_kmeans.png'): continue
        
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        print(f"Processing image: {image_file}")
        
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        means = kmeans_gpu(img)
        cv2.imwrite(image_path.replace(".png", "_kmeans.png"), means)