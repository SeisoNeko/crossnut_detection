import cv2
import numpy as np

# 定義 HSV 上下界
red_lower_hsv  = np.array([0,  186, 139], dtype=np.uint8)
red_upper_hsv  = np.array([10, 255, 209], dtype=np.uint8)

green_lower_hsv = np.array([76, 210, 45], dtype=np.uint8)
green_upper_hsv = np.array([88, 255, 122], dtype=np.uint8)

blue_lower_hsv  = np.array([99,  39,  55], dtype=np.uint8)
blue_upper_hsv  = np.array([118, 255, 244], dtype=np.uint8)

# 將上下界轉換成 BGR 再轉 HLS
def hsv_to_hls_range(lower_hsv, upper_hsv):
    # 建立一張單像素圖像進行轉換
    lower_bgr = cv2.cvtColor(np.uint8([[lower_hsv]]), cv2.COLOR_HSV2BGR)
    upper_bgr = cv2.cvtColor(np.uint8([[upper_hsv]]), cv2.COLOR_HSV2BGR)

    lower_hls = cv2.cvtColor(lower_bgr, cv2.COLOR_BGR2HLS)[0,0]
    upper_hls = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2HLS)[0,0]
    
    return lower_hls, upper_hls

red_hls   = hsv_to_hls_range(red_lower_hsv, red_upper_hsv)
green_hls = hsv_to_hls_range(green_lower_hsv, green_upper_hsv)
blue_hls  = hsv_to_hls_range(blue_lower_hsv, blue_upper_hsv)

print(red_hls)
