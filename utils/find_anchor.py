import cv2
import os
import numpy as np
from cv2.typing import MatLike
from ultralytics import YOLO
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

def check_transparent(image: MatLike, x1: int, y1: int, x2: int, y2: int, tolerance_ratio: float = 0.03) -> bool:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, 1)
    alpha = image[:, :, 3]
    line_alpha = cv2.bitwise_and(alpha, alpha, mask=mask)
    
    # 統計：總像素數 / 不透明像素數
    total_line_pixels = cv2.countNonZero(mask)
    opaque_pixels = cv2.countNonZero(line_alpha)

    # 容忍透明像素數量為圖寬 × ratio，向上取整避免精度誤差
    max_transparent_pixels = int(np.ceil(min(image.shape[:2]) * tolerance_ratio))
    transparent_pixels = total_line_pixels - opaque_pixels

    return transparent_pixels <= max_transparent_pixels


def find_label_anchor(labels: list[MatLike], positions: np.ndarray, confs: np.ndarray, labels_dir: str, device: str) -> tuple[list[tuple[int, int]], bool]:
    labels_height = []
    warning = False

    # K-means clustering
    mean_labels = list(map(kmeans_gpu, labels, [device for _ in labels]))
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

    # #TODO: x,y對調 / 試試C(n,n-1)-->then? 怎麼比較? r^2?
    # x = positions[:,0]
    # y = positions[:,1]
    # plumb_line = poly.Polynomial.fit(x, y, 1)
    
    # r2 = 1 - np.sum((y - plumb_line(x))**2) / np.sum((y - np.mean(y))**2)
    # print("r2: ", r2)
    
    # plt.plot(x, y, 'yo', x, plumb_line(x), '--k')
    # plt.show()

    # remove duplicate blacks
    if labels_height.count("black") > 1:
        warning = True
        indexs = [i for i, color in enumerate(labels_height) if color == "black"]
        max_index = max(indexs, key=lambda x: confs[x])
        for i in indexs:
            if i != max_index:
                labels_height.pop(i)
                confs.pop(i)
                positions = np.delete(positions, i, axis=0)
                
    if labels_height.count("white") > 1:
        warning = True
        indexs = [i for i, color in enumerate(labels_height) if color == "white"]
        max_index = max(indexs, key=lambda x: confs[x])
        for i in indexs:
            if i != max_index:
                labels_height.pop(i)
                confs.pop(i)
                positions = np.delete(positions, i, axis=0)
    
    
    positions_y = positions[:, 1]
    anchors = sorted(list(zip(labels_height, positions_y)), key=lambda x: x[1])
    tmp = []
    for i, (height, y) in enumerate(anchors):
        if height == BLACK_HEIGHT and i != 0:
            warning = True
            continue
        if height == WHITE_HEIGHT and i != len(anchors) - 1:
            warning = True
            continue
        else:
            tmp.append((y, height))

    return tmp, warning


def find_number_anchor(img: MatLike, output_dir: str, img_name: str, model: YOLO) -> list[tuple[int, int]]:
    # check if the output path is valid    # check if the output path is valid
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/num_detect", exist_ok=True)
    

    result = model.predict(cv2.cvtColor(img, cv2.COLOR_BGRA2BGR), project=output_dir, name='num_detect', exist_ok=True, conf=0.3, verbose=False)[0]
    result.save(f"{output_dir}/num_detect/{img_name}_num_detect.png")

    boxes = result.boxes
    cls_ids = boxes.cls.cpu().numpy().astype(int) # 取出 class_id（物件編號 -> "數字幾-1"）
    centers_y = boxes.xywh.cpu().numpy()[:, 1]  # 取中心點 y 座標
    
    positions = []
    values = []
    value_sample = [8, 18, 28, 38, 48, 58, 68, 78, 88] # class_id = 1~9

    for cls_id, y in zip(cls_ids, centers_y):
        if 0 <= cls_id < len(value_sample):  # 確保索引合法
            positions.append(y)
            values.append(value_sample[cls_id])

    return list(zip(positions, values))



if __name__ == "__main__":
    input_dir = './output/final/labels/P_20250430_143423'  # Directory containing cropped images

    labels = []
    for image_file in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_file)  # Full path to the image
        if not image_file.endswith('.png') or image_file.endswith('_kmeans.png') or image_file.startswith('labels'): continue
        labels.append(cv2.imread(image_path))
    
    
    positions = np.array([[1568,1127],
                          [1642,3436],
                          [1629,2995],
                          [1591,1878],
                          [1607,2480]])
    confs = np.array([0.88742, 0.86251, 0.8286, 0.81845, 0.77712])
    output_dir = "output_directory"
 
    find_label_anchor(labels, positions, confs, output_dir)