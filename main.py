import ultralytics
import sys
import os
import time
import cv2
import numpy.polynomial.polynomial as poly
import traceback

from utils.find_cross import find_cross
from utils.find_cross_point import find_cross_point
from utils.find_label import find_label
from utils.find_anchor import find_label_anchor, find_number_anchor
from utils.image_draw_text import draw_text
from utils.upscale import upscale_image
from tkinter import filedialog

if __name__ == '__main__':
    with open('log.txt', 'w', encoding="UTF-8") as log:
        sys.stdout = log

        cross_model = ultralytics.YOLO('model/cross_best_yolo11n-seg.pt')
        label_model = ultralytics.YOLO('model/find_label_best.pt')
        number_model = ultralytics.YOLO('model/number_detection_best.pt')

        input_dir = ''
        # 使用拖曳的方式取得圖片資料夾
        if len(sys.argv) >= 3 and os.path.isdir(sys.argv[2]):
            # 如果是資料夾路徑
            if sys.argv[2].endswith('/'):
                input_dir = sys.argv[2][:-1]
            else:
                input_dir = sys.argv[2]
        else:
            input_dir = filedialog.askdirectory(title="請選擇圖片資料夾")
            if not input_dir:
                print("沒有選擇資料夾，程式結束。")
                sys.exit(0)
            # input_dir = './pics/colored_1'
        
        print("選擇的資料夾是：", input_dir)
        output_dir = os.path.join('./output', input_dir.split('/')[-1]) # output image directory
        temp_dir = os.path.join(output_dir, 'Temp') # temporary image directory
        success_dir = os.path.join(output_dir, 'OK') # result image directory
        warning_dir = os.path.join(output_dir, 'Warning') # result image directory
        failed_dir = os.path.join(output_dir, 'Failed') # failed image directory

        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(warning_dir, exist_ok=True)
        os.makedirs(failed_dir, exist_ok=True)

        # Caculate inference time
        start_time = time.time()
        result = {}

        for image_file in os.listdir(input_dir):
            try:
                if not image_file.endswith(('.jpg', '.png', '.jpeg', '.webp')):
                    continue

                print()    
                print(f"正在處理: {image_file}")
                image_name = image_file.split('.')[0]
                
                ### find cross image from original image
                img_path = os.path.join(input_dir, image_file)
                img = cv2.imread(img_path)
                cross_img = find_cross(img, img_name=image_name, output_dir=temp_dir, model=cross_model)

                # no cross found
                if cross_img is None:
                    print(f"在圖片中找不到十字!")
                    result[image_name] = "no-cross"
                    texted_img = draw_text(img, "no-cross", text_color=(0, 0, 255))
                    cv2.imwrite(f"{failed_dir}/{image_name}.jpg", texted_img)
                    continue

                # cross_img, cross_point, straight_line = temp
                # print(f"Image: {image_file}")
                # print(f"new Cross point: ({cross_point[0]}, {cross_point[1]})")

                ### find label nums and positions
                temp = find_label(cross_img, img_name=image_name, output_dir=temp_dir, model=label_model)
                label_count, label_centers, label_confs, label_imgs = temp
                
                anchors = []
                if label_count > 0:
                    labels_dir = os.path.join(temp_dir, 'labels', image_name)
                    # anchors, warning = find_label_anchor(label_imgs, label_centers, label_confs, labels_dir=labels_dir, device=sys.argv[1])
                    anchors = find_label_anchor(label_imgs, label_centers, label_confs, labels_dir=labels_dir, device=sys.argv[1])
                    print(f"找到 {len(anchors)} 個標籤")

                if len(anchors) <= 2: # can't use interpolation (插值)
                    print(f"標籤數量無法計算高度: {len(anchors)}")
                    warning = True
                    # cross_path = os.path.join(temp_dir, 'crops', f"{image_name}_cross.png")
                    # upscale_image(cross_path, f"{temp_dir}/upscaled") # 放大圖片
                    # cross_img = cv2.imread(f"{temp_dir}/upscaled/{image_name}_cross.png")
                    number_anchor = find_number_anchor(cross_img, temp_dir, image_name, number_model)
                    anchors.extend(number_anchor)
                    print(f"找到 {len(number_anchor)} 個數字錨點")

                if len(anchors) <= 2: # still can't use interpolation
                    print(f"找不到足夠的錨點來計算高度: {len(anchors)}")
                    result[image_name] = "not-enouth-anchors"
                    texted_img = draw_text(img, "not-enouth-anchors", text_color=(0, 0, 255))
                    cv2.imwrite(f"{failed_dir}/{image_name}.jpg", texted_img)
                    continue
                
                ### find cross point
                cross_point = find_cross_point(cross_img, temp_dir, image_name)

                # no cross point found
                if cross_point == None:
                    print("找不到十字交點!")
                    result[image_name] = "no-cross-point"
                    texted_img = draw_text(img, "no-cross-point", text_color=(0, 0, 255))
                    cv2.imwrite(f"{failed_dir}/{image_name}.jpg", texted_img)
                    continue

                cross_x, cross_y = cross_point
                print(f"交點座標: ({cross_x}, {cross_y})")
                
                
                ### 建立擬合模型
                pixels_y, heights = zip(*anchors)
                height_estimator = poly.Polynomial.fit(pixels_y, heights, 2)
                estimated_height = height_estimator(cross_y)
                if warning:
                    print(f"估計高度(警告): {estimated_height:.1f}cm")
                    result[image_name] = f"*{estimated_height:.1f}cm*"
                    texted_img = draw_text(img, f"*{estimated_height:.1f}cm*", text_color=(0, 0, 255))
                    cv2.imwrite(f"{warning_dir}/{image_name}.jpg", texted_img)
                else:
                    print(f"估計高度: {estimated_height:.1f}cm")
                    result[image_name] = f"{estimated_height:.1f}cm"
                    texted_img = draw_text(img, f"{estimated_height:.1f}cm")
                    cv2.imwrite(f"{success_dir}/{image_name}.jpg", texted_img)

                
            except Exception as e:
                print(f"Error processing image {image_file}: {e}")
                result[image_name] = "error"
                texted_img = draw_text(img, "error", text_color=(0, 0, 255))
                cv2.imwrite(f"{failed_dir}/{image_name}.jpg", texted_img)
                # traceback.print_exc()
                continue

        ### Caculate inference time
        end_time = time.time()
        inference_time = end_time - start_time
        print(f"Inference time: {inference_time:.2f} seconds")
        
        ### Save results to a csv file

        success_count = 0
        warning_count = 0
        failed_count = 0

        result_file = os.path.join(output_dir, 'results.csv')
        with open(result_file, 'w') as f:
            f.write("Image, Result\n")
            for img_name, res in result.items():
                f.write(f"{img_name}, {res}\n")

                if res.endswith("cm"):
                    success_count += 1
                elif res.startswith("*") and res.endswith("*"):
                    warning_count += 1
                else:
                    failed_count += 1

            # 寫入統計數字
            f.write("\n")
            f.write(f"Success:, {success_count}\n")
            f.write(f"Warning:, {warning_count}\n")
            f.write(f"Failed:, {failed_count}\n")
