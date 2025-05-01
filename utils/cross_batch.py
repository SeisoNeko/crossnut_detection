import cv2
import numpy as np
import glob
import os

# 找交點
def line_intersection(line1, line2):
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

    return int(px), int(py)



# 讀取圖片
input_dir = "../colored_label/output/crops"
output_dir = "./output_color"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_paths = glob.glob(os.path.join(input_dir, "*.jpg"))

for img_path in image_paths:
    # 讀取一張圖
    img = cv2.imread(img_path)

    # 灰階
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray_large = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("gray", gray_large)

    # 二值化
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # thresh_large = cv2.resize(thresh, (thresh.shape[1]*2, thresh.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("thresh", thresh_large)

    # 霍夫直線轉換
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # 儲存直線座標
    line_coords = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_coords.append(((x1, y1), (x2, y2)))
        
        # 複製一張乾淨的底圖來畫線（避免每次都畫在同一張上）
        # img_copy = img.copy()
        # cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # 嘗試所有線段組合找交點
    points = []
    for i in range(len(line_coords)):
        for j in range(i+1, len(line_coords)):
            pt = line_intersection(line_coords[i], line_coords[j])
            if pt:
                x, y = pt
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    points.append((x, y))
                    img[y, x] = (0, 0, 128)  # 直接把這個點設成紅色
                    # print("交點座標:", (x, y))


    # 排除找不到交點的情況
    if len(points) == 0:
        print(f"Warning: no intersection points found for {img_path}, skipping.")
        continue  # 跳過這張圖，不要再做後面的處理

    ### 排除極端值
    # 轉成 numpy array
    points_arr = np.array(points)

    # 計算中位數中心點
    median_x = np.median(points_arr[:,0])
    median_y = np.median(points_arr[:,1])

    max_dist = 50

    # 篩選出距離中位數中心不超過 max_dist 的點
    good_points = []
    for (x, y) in points:
        dist = np.sqrt((x - median_x)**2 + (y - median_y)**2)
        if dist < max_dist:
            good_points.append((x, y))
            img[y, x] = (255, 0, 0)

    ### 找最終交點位置
    good_points_arr = np.array(good_points)

    # 分別取 x 跟 y 的平均
    mean_x = np.mean(good_points_arr[:, 0])
    mean_y = np.mean(good_points_arr[:, 1])

    # 四捨五入取整數（因為 pixel 座標要是整數）
    final_x = int(round(mean_x))
    final_y = int(round(mean_y))

    cv2.circle(img, (final_x, final_y), 2, (0, 255, 0), -1)
    # img[final_y, final_x] = (0, 255, 0)
    cv2.putText(img, f"({final_x},{final_y})", (final_x + 20, final_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # print("加權平均後代表座標:", (final_x, final_y))

    # 顯示結果
    # img_large = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_NEAREST)
    # cv2.imshow("Result2", img_large)

    # 把檔名取出來，不要路徑
    filename = os.path.basename(img_path)
    
    # 設定輸出路徑
    output_path = os.path.join(output_dir, filename)

    # 存圖
    # cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 100])

    output_path = os.path.splitext(output_path)[0] + ".png"
    cv2.imwrite(output_path, img)

    # cv2.imshow(f"{output_path}", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
