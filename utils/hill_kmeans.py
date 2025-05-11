import cupy as cp
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from cv2.typing import MatLike

yellow_lower = np.array([20, 80, 50], np.uint8)
yellow_upper = np.array([35, 255, 255], np.uint8)

def hill_climbing_algorithm_flat(flat_rgb, num_bins):
    """
    Implements the hill-climbing algorithm for a flat RGB array.

    Parameters:
        flat_rgb (numpy.ndarray): Input flat RGB array of shape (num_pixels, 3).
        num_bins (int): Number of bins per channel.

    Returns:
        peaks (list): List of peaks found in the image.
    """
    # Step 1: Initialization
    A = np.zeros((num_bins, num_bins, num_bins), dtype=int)

    # Normalize pixel values to fit into the bins
    normalized_rgb = (flat_rgb / 256 * num_bins).astype(int)
    for r, g, b in normalized_rgb:
        A[r, g, b] += 1

    # Step 2: Find peaks of A
    peaks = []
    for i in range(1, num_bins - 1):
        for j in range(1, num_bins - 1):
            for k in range(1, num_bins - 1):
                # Check if A(i, j, k) is greater than all its neighbors
                neighbors = A[i-1:i+2, j-1:j+2, k-1:k+2].flatten()
                neighbors = neighbors[neighbors != A[i, j, k]]  # Exclude the current bin value
                if neighbors.size > 0 and A[i, j, k] > np.max(neighbors):
                    # Average the intensity values of pixels in the (i, j, k) bin
                    peak_value = np.mean(flat_rgb[(normalized_rgb[:, 0] == i) &
                                                  (normalized_rgb[:, 1] == j) &
                                                  (normalized_rgb[:, 2] == k)], axis=0)
                    peaks.append(peak_value)

    # Step 3: Output peaks
    return peaks

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
def error_iteration(centroids, k, Y, N, epsilon, batch_size=500):
    """
    Performs k-means clustering with batch processing to reduce memory usage.

    Parameters:
        centroids (cupy.ndarray): Initial centroids of shape (3, k).
        k (int): Number of clusters.
        batch_size (int): Number of pixels to process in each batch.

    Returns:
        centroids (cupy.ndarray): Final centroids of shape (3, k).
        cluster_assignments (cupy.ndarray): Cluster assignments for each pixel.
    """
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


def hill_kmeans(img: MatLike) -> MatLike:
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
    k = 4  # Number of clusters for the first k-means
    k2 = 3  # Number of clusters for the second k-means
    epsilon = 5  # Stopping condition
    batch_size = 1000  # Batch size for distance computation

    # Image dimensions
    original_height, original_width = img.shape[:2]

    num_bins = 16  # Number of bins for histogram
    
    
    # Step 1. Run the hill-climbing algorithm on filtered_rgb
    print("Running hill-climbing algorithm...")
    peaks = hill_climbing_algorithm_flat(filtered_rgb, num_bins)
    print(f"Number of peaks found: {len(peaks)}")
    print(f"Peaks: {peaks}")

    k = len(peaks)  # Number of clusters based on peaks

    # Step 2. Use K-Means
    print("Running K-means...")

    init_centroids = cp.zeros((3, k))
    init_centroids[:, 0] = Y[:, cp.random.randint(0, N)]  # First centroid is random

    centroids = color_Iteration(init_centroids, k, Y)  # Initialize centroids using farthest point method

    # Perform k-means clustering
    centroids, cluster_assignments = error_iteration(centroids, k, Y, N, epsilon, batch_size)

    # Reconstruct the full image with clustered colors
    clustered_flat_rgb = cp.asnumpy(centroids[:, cluster_assignments].T.astype(np.uint8))  # Convert back to NumPy
    reconstructed_rgb = np.zeros((flat_rgb.shape[0], 3), dtype=np.uint8)  # Initialize with zeros
    reconstructed_rgb[non_transparent_mask] = clustered_flat_rgb  # Map clustered colors to non-transparent pixels


    # Combine with the original alpha channel
    reconstructed_image = np.zeros((flat_alpha.size, 3), dtype=np.uint8)  # RGBA
    reconstructed_image[:, :3] = reconstructed_rgb  # Add RGB values
    # reconstructed_image[:, 3] = flat_alpha  # Add the original alpha channel

    # Reshape back to the original image dimensions
    reconstructed_image = reconstructed_image.reshape(original_height, original_width, 3)

    return cv2.cvtColor(reconstructed_image, cv2.COLOR_RGB2BGR)

if __name__ == "__main__":
    # Load the image with transparency (RGBA)
    img = cv2.imread('img/crossline.jpg', cv2.IMREAD_UNCHANGED)
    
    ret = hill_kmeans(img)

    cv2.imwrite("clustered_image_with_alpha.png", ret)