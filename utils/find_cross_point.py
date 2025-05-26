import cv2
import numpy as np
import os
import time

from cv2.typing import MatLike

# 克拉馬找交點
def line_intersection(line1: tuple[int, int], line2: tuple[int, int]) -> tuple[int, int] | None:
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    a1 = x2 - x1
    b1 = -(x4 - x3)
    a2 = y2 - y1
    b2 = -(y4 - y3)
    c1 = x3 - x1
    c2 = y3 - y1

    # 克拉馬公式解
    denom = a1 * b2 - a2 * b1
    if denom == 0:
        return None  # 平行或重合
    
    # 找出 t1
    det_m1 = c1 * b2 - c2 * b1
    t1 = det_m1 / denom

    # 代回 t1 計算真正交點
    px = x1 + t1 * (x2 - x1)
    py = y1 + t1 * (y2 - y1)

    # return int(px), int(py)
    return px, py

# ---

def find_cross_point(image: MatLike, img_name: str, output_dir: str) -> tuple[int, int]:
    # check if the output path is valid
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'cross_point')):
        os.makedirs(os.path.join(output_dir, 'cross_point'))

    start_time = time.time()

    # 轉換成 BGR 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) 

    # 取出高度與寬度
    height, width = image.shape[:2]

    # 霍夫直線轉換 TODO: 修改成改變解析度
    ########### Deprecated
    # if height > 1000 and width > 1000:
    #     threshold = int(max(height, width) * 0.4)
    #     minLineLength = threshold
    #     maxLineGap = int(max(height, width) * 0.05)
    # else:
    threshold = 100
    minLineLength = 100
    maxLineGap = 10

    # 依據解析度改變 scale
    scale_x = 1
    scale_y = 1
    while height > 1000:
        scale_y /= 2
        height /= 2
    while height < 500:
        scale_y *= 2
        height *= 2
    while width > 1000:
        scale_x /= 2
        width /= 2
    while width < 500:
        scale_x *= 2
        width *= 2

    scaled_img = cv2.resize(image, None, fx=scale_x, fy=scale_y)

    ### debug
    # print(f"scale_y: {scale_y}, scale_x: {scale_x}")
    # cv2.imshow("scaled_img", scaled_img)
    # cv2.waitKey(0)  # 等待按鍵，按一次才會跳下一張
    # cv2.destroyWindow(f"{i}")  # 顯示完關掉目前這個小視窗

    # 灰階
    gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)

    # 二值化（threshold 設為 1）
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=threshold, minLineLength=minLineLength, maxLineGap=maxLineGap) 

    # 儲存直線座標
    line_coords = []
    # i = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_coords.append(((x1, y1), (x2, y2)))
        
        ### debug
        # 複製一張乾淨的底圖來畫線（避免每次都畫在同一張上）
        # img_copy = scaled_img.copy()
        # cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

        # 逐個直線顯示（用順序當作視窗名稱）-> uncomment i = 0
        # img_copy_large = cv2.resize(img_copy, (img_copy.shape[1]*2, img_copy.shape[0]*2), interpolation=cv2.INTER_NEAREST)
        # cv2.imshow(f"{i}", img_copy)
        # cv2.waitKey(0)  # 等待按鍵，按一次才會跳下一張
        # cv2.destroyWindow(f"{i}")  # 顯示完關掉目前這個小視窗
        # i = i + 1

    # print(f"有 {i} 個線")

    # 嘗試所有線段組合找交點
    points = []
    for i in range(len(line_coords)):
        for j in range(i+1, len(line_coords)):
            pt = line_intersection(line_coords[i], line_coords[j])
            if pt:
                x, y = pt
                x = int(x / scale_x)
                y = int(y / scale_y)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    points.append((x, y))
                    image[y, x] = (0, 0, 255)  # 把這個點設成紅色
                    # print("交點座標:", (x, y))

    print(f"有 {len(points)} 個點")

    ### 排除極端值
    points_arr = np.array(points)

    # 計算中位數中心點
    median_x = np.median(points_arr[:,0])
    median_y = np.median(points_arr[:,1])

    # max_dist = 50
    std_x = np.std(points_arr[:, 0])  # x 座標的標準差
    std_y = np.std(points_arr[:, 1])  # y 座標的標準差
    avg_std = int((std_x + std_y) / 2)
    max_dist = avg_std if avg_std > 50 else 50

    # 篩選出距離中位數中心不超過 max_dist 的點
    good_points = []
    for (x, y) in points:
        dist = np.sqrt((x - median_x)**2 + (y - median_y)**2)
        if dist < max_dist:
            good_points.append((x, y))
            image[y, x] = (255, 0, 0)

    ### 找最終交點位置
    good_points_arr = np.array(good_points)

    # 分別取 x 跟 y 的平均
    mean_x = np.mean(good_points_arr[:, 0])
    mean_y = np.mean(good_points_arr[:, 1])

    # 四捨五入（因為 pixel 座標要是整數）
    final_x = int(np.round(mean_x))
    final_y = int(np.round(mean_y))

    cv2.circle(image, (final_x, final_y), 2, (0, 255, 0), -1)
    # img[final_y, final_x] = (0, 255, 0)
    cv2.putText(image, f"({final_x},{final_y})", (final_x + 20, final_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    """
    print("加權平均後代表座標:", (final_x, final_y))

    # 顯示結果
    img_large = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("Result2", img_large) # 放大兩倍
    cv2.imshow("Result", img) # 原圖
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    cv2.imwrite(f"{output_dir}/cross_point/{img_name}_crosspoint.png", image)

    end_time = time.time()

    print(f"Find cross point total time: {(end_time - start_time) * 1000}ms")

    return (final_x, final_y)


if __name__ == "__main__":
    import os
    # input_dir = './temp/crops'  # Directory containing cropped images
    output_dir = '../output/temp'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_name = "PXL_20250415_094502342_cross.png"
    img_path = "../output/labeled_cross_photos/crops/PXL_20250415_094502342_cross.png"
    img = cv2.imread(img_path)
    find_cross_point(img, img_name, output_dir)

    # for img_file in os.listdir(input_dir):
    #     img_name = img_file.split('.')c[0]
    #     img_path = os.path.join(input_dir, img_file)
    #     img = cv2.imread(img_path)
    #     find_cross_point(img, img_name, output_dir)