import ultralytics
import os
from utils.find_cross import find_cross
from utils.find_label import find_label

if __name__ == '__main__':
    cross_model = ultralytics.YOLO('model/cross_best.pt')
    find_label_model = ultralytics.YOLO('model/find_label_best.pt')
    number_detection_model = ultralytics.YOLO('model/number_detection_best.pt')

    input_dir = './data/newest'     # input image directory, change as your wish
    output_dir = './output/final' # output image directory

    for image_file in os.listdir(input_dir):
        try:
            image_path = os.path.join(input_dir, image_file)
            find_cross(image_path=image_path, output_dir='./temp', model=cross_model)

            cross_image_name = image_file.split('.')[0] + '_crossline.png'
            cross_image_path = os.path.join('./temp/crops', cross_image_name)
            numbers_of_label, label_positions = find_label(image_path=cross_image_path, output_dir='./temp', model=find_label_model)
            print(f"Image: {cross_image_name}, Number of labels: {numbers_of_label}, Positions: {label_positions}")
            
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue
