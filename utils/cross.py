import cv2
import numpy as np

# 克拉馬找交點
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

# ---

# 讀取圖片
img = cv2.imread("../crossnut_detection/output_gu/crops/273_crossline.jpg")

# 灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_large = cv2.resize(gray, (gray.shape[1]*2, gray.shape[0]*2), interpolation=cv2.INTER_NEAREST)
cv2.imshow("gray", gray_large)

# 二值化（threshold 設為 1）
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
thresh_large = cv2.resize(thresh, (thresh.shape[1]*2, thresh.shape[0]*2), interpolation=cv2.INTER_NEAREST)
cv2.imshow("thresh", thresh_large)

# 霍夫直線轉換
lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 儲存直線座標
line_coords = []
# i = 0
for line in lines:
    x1, y1, x2, y2 = line[0]
    line_coords.append(((x1, y1), (x2, y2)))
    
    # 複製一張乾淨的底圖來畫線（避免每次都畫在同一張上）
    img_copy = img.copy()
    cv2.line(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 1)
    
    # 逐個直線顯示（用順序當作視窗名稱）-> uncomment i = 0
    # cv2.imshow(f"{i}", img_copy)
    # cv2.waitKey(0)  # 等待按鍵，按一次才會跳下一張
    # cv2.destroyWindow(f"{i}")  # 顯示完關掉目前這個小視窗
    # i = i + 1

# 嘗試所有線段組合找交點
points = []
for i in range(len(line_coords)):
    for j in range(i+1, len(line_coords)):
        pt = line_intersection(line_coords[i], line_coords[j])
        if pt:
            x, y = pt
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                points.append((x, y))
                img[y, x] = (0, 0, 255)  # 把這個點設成紅色
                print("交點座標:", (x, y))

### 排除極端值
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

# 四捨五入（因為 pixel 座標要是整數）
final_x = int(round(mean_x))
final_y = int(round(mean_y))

cv2.circle(img, (final_x, final_y), 2, (0, 255, 0), -1)
# img[final_y, final_x] = (0, 255, 0)
cv2.putText(img, f"({final_x},{final_y})", (final_x + 20, final_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

print("加權平均後代表座標:", (final_x, final_y))

# 顯示結果
img_large = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2), interpolation=cv2.INTER_NEAREST)
cv2.imshow("Result2", img_large) # 放大兩倍
cv2.imshow("Result", img) # 原圖
cv2.waitKey(0)
cv2.destroyAllWindows()
