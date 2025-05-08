import ultralytics
import os
import time
import cv2

from utils.find_cross import find_cross
from utils.find_label import find_label
from utils.find_cross_point import find_cross_point
from utils.find_scale import find_scale

if __name__ == '__main__':
    cross_model = ultralytics.YOLO('model/cross_best.pt')
    find_label_model = ultralytics.YOLO('model/find_label_best.pt')
    number_detection_model = ultralytics.YOLO('model/number_detection_best.pt')

    # input_dir = './pics/1140430-gov2'   # input image directory, change as your wish
    input_dir = './pics/colored_1'   # input image directory, change as your wish
    output_dir = './output/final' # output image directory

    # Caculate inference time
    start_time = time.time()

    for image_file in os.listdir(input_dir):
        try:
            # find cross image from original image
            img_path = os.path.join(input_dir, image_file)
            img = cv2.imread(img_path)

            cross_img = find_cross(img, img_name=image_file.split('.')[0], output_dir='./temp', model=cross_model)

            # no cross found
            if cross_img is None:
                continue

            # find label nums and positions
            # cross_img_name = image_file.split('.')[0] + '_cross.png' # _crossline.png for cross_best_old.pt
            # cross_img_path = os.path.join('./temp/crops', cross_img_name)

            temp = find_label(cross_img, img_name=image_file.split('.')[0], output_dir='./temp', model=find_label_model)
            label_count, label_positions, label_confs, label_imgs = temp
            
            # print(f"Image: {cross_img_name}, Number of labels: {label_count} \nPositions: \n{label_positions}")

            # find cross point
            # cross_point = find_cross_point(cross_img_path)
            # print(f"Cross point: {cross_point}")

            
            if label_count <= 2: # can't use interpolation (插值)
                # TODO: add a function to find the label position
                pass
            elif label_count > 5: # more than 5 labels (illigal case)
                # TODO: check the illegal label
                pass
            else: # calculate scale of labels
                labels_dir = os.path.join('./temp/labels', image_file.split('.')[0])
                poly = find_scale(label_imgs, label_positions, label_confs, labels_dir=labels_dir)


        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

    # Caculate inference time
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")