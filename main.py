import ultralytics
import os
import time
from utils.find_cross import find_cross
from utils.find_label import find_label
from utils.find_cross_point import find_cross_point

if __name__ == '__main__':
    cross_model = ultralytics.YOLO('model/cross_best.pt')
    find_label_model = ultralytics.YOLO('model/find_label_best.pt')
    number_detection_model = ultralytics.YOLO('model/number_detection_best.pt')

    input_dir = './data/newest'     # input image directory, change as your wish
    output_dir = './output/final' # output image directory

    # Caculate inference time
    start_time = time.time()

    for image_file in os.listdir(input_dir):
        try:
            #find cross image from original image
            image_path = os.path.join(input_dir, image_file)
            find_cross(image_path=image_path, output_dir='./temp', model=cross_model)

            #find label nums and positions
            cross_image_name = image_file.split('.')[0] + '_crossline.png'
            cross_image_path = os.path.join('./temp/crops', cross_image_name)
            numbers_of_label, label_positions = find_label(image_path=cross_image_path, output_dir='./temp', model=find_label_model)
            print(f"Image: {cross_image_name}, Number of labels: {numbers_of_label}, Positions: {label_positions}")

            #find cross point
            cross_point = find_cross_point(cross_image_path)
            print(f"Cross point: {cross_point}")

        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue

    # Caculate inference time
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time:.2f} seconds")