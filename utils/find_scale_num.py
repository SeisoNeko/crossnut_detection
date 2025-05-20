import cv2
import numpy as np
import os
from cv2.typing import MatLike
from ultralytics import YOLO
# import matplotlib.pyplot as plt

def find_scale_num(img: MatLike, output_dir: str, img_name: str, model: YOLO, cross_point: tuple[int, int]) -> float:
    '''
    # 模擬輸入資料（例如 y 座標 與 對應數字刻度）
    list_y = [100, 200, 300, 400] # 這是刻度
    list_z = [0, 1, 2.3, 3.7]  # 對應的實際數值

    # 用二次多項式進行擬合（你也可以改成 1 做線性）
    coeffs = np.polyfit(list_y, list_z, 2)
    poly = np.poly1d(coeffs)

    # 輸出結果
    print("poly 就是你擬合出來的模型：")
    print(poly)
    print(poly(350))
    '''

    # check if the output path is valid
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'num_detect')):
        os.makedirs(os.path.join(output_dir, 'num_detect'))
    if not os.path.exists(os.path.join(output_dir, 'reading')):
        os.makedirs(os.path.join(output_dir, 'reading'))
    
    cross_point_x, cross_point_y = cross_point

    result = model.predict(img, project=output_dir, name='num_detect', exist_ok=True, conf=0.3, verbose=False)[0]
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

    if len(positions) >= 2:
        poly = np.poly1d(np.polyfit(positions, values, deg=2))  # 建立擬合模型
    else:
        raise ValueError("偵測的數字點不足以擬合模型")

    # print(f"交點座標： {cross_point_y}")
    # print(f"cls_ids: {cls_ids}")
    # print(f"centers_y: {centers_y}")
    # print(f"positions: {positions}")
    # print(f"values: {values}")


    # print("擬合出來的模型 poly：")
    # print(poly)
    # print(f"交點讀數： {poly(cross_point_y)}")
    
    reading = poly(cross_point_y)
    height, width = img.shape[:2]

    if height > 1000 and width > 1000:
        cv2.circle(img, (cross_point_x, cross_point_y), 10, (0, 255, 0), -1)
        cv2.putText(img, f'{reading:.2f} cm', (cross_point_x + 80, cross_point_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10, lineType=cv2.LINE_AA)
    else:
        cv2.circle(img, (cross_point_x, cross_point_y), 2, (0, 255, 0), -1)
        cv2.putText(img, f'{reading:.2f} cm', (cross_point_x + 20, cross_point_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    cv2.imwrite(f"{output_dir}/reading/{img_name}_reading.png", img)

    return reading

if __name__ == "__main__":
    import os
    import ultralytics
    input_path = '../output/final/crops/PXL_20250415_093754629_cross.png'
    img_file = 'PXL_20250415_093754629_cross.png'
    img = cv2.imread(input_path)
    output_dir = '../output/temp'
    # cross_point_y = 1224
    cross_point = 1672, 1224
    number_detection_model = ultralytics.YOLO('../model/number_detection_best.pt')

    reading = find_scale_num(img, output_dir, img_file.split('.')[0], number_detection_model, cross_point)