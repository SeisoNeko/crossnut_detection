import cv2
import numpy as np
from cv2.typing import MatLike


def find_scale(labels: list[MatLike], positions: np.ndarray, confs: np.ndarray, labels_dir: str) -> np.poly1d:

    # Set range for red color 
    red_lower1 = np.array([0,   88,  146], dtype=np.uint8)
    red_upper1 = np.array([10,  105, 255], dtype=np.uint8)
    red_lower2 = np.array([167, 73,  126], dtype=np.uint8)
    red_upper2 = np.array([179, 120, 255], dtype=np.uint8)

    #green color
    green_lower = np.array([50, 11, 158], dtype=np.uint8)
    green_upper = np.array([93, 76, 255], dtype=np.uint8)

    #blue color
    blue_lower = np.array([96,  36,  0  ], dtype=np.uint8)
    blue_upper = np.array([123, 137, 255], dtype=np.uint8)
        
    # K-means clustering
    mean_labels = list(map(kmeans_gpu, labels))
    for i, img in enumerate(mean_labels):
        cv2.imwrite(f"{labels_dir}/label_{i}_kmeans.png", img)
        


        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        # Create masks for each color
        red_mask = cv2.inRange(hls, red_lower1, red_upper1) | cv2.inRange(hls, red_lower2, red_upper2)
        green_mask = cv2.inRange(hls, green_lower, green_upper)
        blue_mask = cv2.inRange(hls, blue_lower, blue_upper)
        
        cv2.imshow(f"label_{i}", img)
        cv2.imshow(f"red", red_mask)
        cv2.imshow(f"green", green_mask)
        cv2.imshow(f"blue", blue_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
        
        rgb_mask = red_mask | green_mask | blue_mask

        # 黑白分析
        non_rgb_mask = cv2.bitwise_not(rgb_mask)
        hls_filtered = hls[non_rgb_mask > 0]
        if len(hls_filtered) > 0:
            h, l, s = hls_filtered[:,0], hls_filtered[:,1], hls_filtered[:,2]
            black_pixels = np.sum((l < 50) & (s > 5))
            white_pixels = np.sum((l > 120) & (s < 10))
        else:
            black_pixels = 0
            white_pixels = 0

        counts = {
            "red":   cv2.countNonZero(red_mask),
            "green": cv2.countNonZero(green_mask),
            "blue":  cv2.countNonZero(blue_mask),
            "black": black_pixels,
            "white": white_pixels
        }

        print(max(counts, key=counts.get))

    print()
    # poly = np.poly1d(np.polyfit(list_y, list_z, 2))
    # print(poly)
    # print("Estimated distance: ", poly(gray_y))
    # plt.plot(list_y, list_z, 'yo', list_y, poly(list_y), '--k')
    # plt.show()



if __name__ == "__main__":
    from kmeansgpu import kmeans_gpu
    import os
    input_dir = './temp/labels/PXL_20250415_094322101'  # Directory containing cropped images

    labels = []
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        if not image_file.endswith('.png') or image_file.endswith('_kmeans.png'): continue
        labels.append(cv2.imread(image_path))
    
    
    positions = np.array([[1568,1024,1773,1127],
                          [1642,3378,1793,3436],
                          [1629,2934,1790,2995],
                          [1591,1793,1784,1878],
                          [1607,2409,1779,2480]])
    confs = np.array([0.88742, 0.86251, 0.8286, 0.81845, 0.77712])
    output_dir = "output_directory"
 
    find_scale(labels, positions, confs, output_dir)
else:
    from .kmeansgpu import kmeans_gpu