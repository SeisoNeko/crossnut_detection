import ultralytics
import os
import time
import cv2
import numpy as np
import traceback

from utils.find_cross import find_cross
from utils.find_label import find_label
from utils.find_cross_point import find_cross_point
from utils.find_anchor import find_label_anchor, find_number_anchor

if __name__ == '__main__':
    cross_model = ultralytics.YOLO('model/cross_best_yolo11n-seg.pt')
    label_model = ultralytics.YOLO('model/find_label_best.pt')
    number_model = ultralytics.YOLO('model/number_detection_best.pt')

    # input_dir = './pics/data'   # input image directory, change as your wish
    # input_dir = './pics/1140430-gov2'   # input image directory, change as your wish
    input_dir = '../photos/labeled_cross_photos'   # input image directory, change as your wish
    output_dir = './output/final/labeled_cross_photos2' # output image directory

    # Caculate inference time
    start_time = time.time()

    for image_file in os.listdir(input_dir):
        try:
            if not image_file.endswith(('.jpg', '.png', '.jpeg', '.webp')):
                continue
            
            print(f"Processing image: {image_file}")
            image_name = image_file.split('.')[0]
            
            ### find cross image from original image
            img_path = os.path.join(input_dir, image_file)
            img = cv2.imread(img_path)
            cross_img = find_cross(img, img_name=image_name, output_dir=output_dir, model=cross_model)

            # no cross found
            if cross_img is None:
                print(f"No cross found in image {image_file}.")
                continue

            # cross_img, cross_point, straight_line = temp
            # print(f"Image: {image_file}")
            # print(f"new Cross point: ({cross_point[0]}, {cross_point[1]})")

            ### find label nums and positions
            temp = find_label(cross_img, img_name=image_name, output_dir=output_dir, model=label_model)
            label_count, label_centers, label_confs, label_imgs = temp
            
            print(f"Number of labels: {label_count}")

            ### find cross point
            cross_point = find_cross_point(cross_img, image_name, output_dir)

            # no cross point found
            if cross_point == None:
                print("No HoughLinesP intersection found.")
                continue

            cross_x, cross_y = cross_point
            print(f"Cross point: ({cross_x}, {cross_y})")

            labels_dir = os.path.join(output_dir, 'labels', image_name)
            anchors = find_label_anchor(label_imgs, label_centers, label_confs, labels_dir=labels_dir)

            if anchors.shape[0] <= 2: # can't use interpolation (插值)
                number_anchor = find_number_anchor(cross_img, output_dir, image_name, number_model)
                anchors = np.concatenate((anchors, number_anchor), axis=0)
            
            ### 建立擬合模型
            poly = np.poly1d(np.polyfit(anchors[:, 1], anchors[:, 0], 2))
            
            print("Estimate height: ", poly(cross_y))
            print()

            

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            print(traceback.print_exc())
            continue

    # Caculate inference time
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")