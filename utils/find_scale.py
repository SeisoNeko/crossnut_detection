import cv2
import numpy as np
import matplotlib.pyplot as plt
from cv2.typing import MatLike
if __name__ == "__main__":
    from kmeansgpu import kmeans_gpu
else:
    from .kmeansgpu import kmeans_gpu

WHITE_HEIGHT = 22
RED_HEIGHT = 42
BLUE_HEIGHT = 62
GREEN_HEIGHT = 82
BLACK_HEIGHT = 102

def is_sorted(l: list[int]) -> bool:
    """
    Check if a list is sorted in ascending order.
    
    Args:
        l (list): The list to check.
        
    Returns:
        bool: True if the list is sorted, False otherwise.
    """
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1))


def find_scale(labels: list[MatLike], positions: np.ndarray, confs: np.ndarray, labels_dir: str) -> np.poly1d:
    labels_height = []

    # K-means clustering
    mean_labels = list(map(kmeans_gpu, labels))
    for i, img in enumerate(mean_labels):
        cv2.imwrite(f"{labels_dir}/label_{i}_kmeans.png", img)

        # cv2.imshow(f"label_{i}", img)
        # cv2.moveWindow(f"label_{i}", 750, 700)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        
        # 分離通道
        bgr = img[:, :, :3]     # (H, W, 3)
        alpha = img[:, :, 3]    # (H, W)

        # 建立非透明遮罩（alpha > 0）
        mask = alpha > 0        # bool mask

        # 將 BGR reshape 成 (N, 3)，mask 成 (N,)
        bgr_flat = bgr.reshape(-1, 3)
        mask_flat = mask.flatten()

        # 取出非透明像素
        valid_pixels = bgr_flat[mask_flat]
        bgr = valid_pixels.mean(axis=0)
                
        hls = cv2.cvtColor(valid_pixels[np.newaxis, :, :], cv2.COLOR_BGR2HLS)[0].mean(axis=0)
        h, l, s = hls
        
            
        # if max(bgr)-min(bgr) < 35: #black/white
        #     if l < 80:
        #         print("black")
        #     else:
        #         print("white")
        if s < 70 or l > 150 or max(bgr)-min(bgr) < 20: # black/white
            if l <= 72:
                labels_height.append(BLACK_HEIGHT)
                # print("black")
            else:
                labels_height.append(WHITE_HEIGHT)
                # print("white")
        elif h < 15 or h > 170:
            labels_height.append(RED_HEIGHT)
            # print("red")
        elif h > 60 and h < 95:
            labels_height.append(GREEN_HEIGHT)
            # print("green")
        elif h > 95 and h < 130:
            labels_height.append(BLUE_HEIGHT)
            # print("blue")
        else:
            labels_height.append(0)
            # print(f"--unknown--{labels_dir}")

    # print()


    
    # remove duplicate blacks
    if labels_height.count("black") > 1:
        indexs = [i for i, color in enumerate(labels_height) if color == "black"]
        max_index = max(indexs, key=lambda x: confs[x])
        for i in indexs:
            if i != max_index:
                labels_height.pop(i)
                confs.pop(i)
                positions = np.delete(positions, i, axis=0)

                
    # # check if the order of y and color is matched
    # datas.sort(key=lambda x: x[1])  # Sort by y coordinate
    # if not is_sorted([d[0] for d in datas]):
    #     print("Error: y coordinates are not sorted.")
    #     return None
    
    
    positions_y = positions.transpose()[1]
    
    poly = np.poly1d(np.polyfit(positions_y, labels_height, 2))
    
    print(poly)
    # plt.plot(positions_y, labels_height, 'yo', positions_y, poly(positions_y), '--k')
    
    return poly




if __name__ == "__main__":
    import os
    input_dir = './output/final/labels/P_20250430_143423'  # Directory containing cropped images

    labels = []
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        if not image_file.endswith('.png') or image_file.endswith('_kmeans.png'): continue
        labels.append(cv2.imread(image_path))
    
    
    positions = np.array([[1568,1127],
                          [1642,3436],
                          [1629,2995],
                          [1591,1878],
                          [1607,2480]])
    confs = np.array([0.88742, 0.86251, 0.8286, 0.81845, 0.77712])
    output_dir = "output_directory"
 
    find_scale(labels, positions, confs, output_dir)
else:
    from .kmeansgpu import kmeans_gpu