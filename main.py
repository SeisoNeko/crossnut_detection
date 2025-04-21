import ultralytics
import os

if __name__ == '__main__':
    model = ultralytics.YOLO('model/color_detect.pt')
    data_dir = 'data/labeled' # Path to the images wanted to inference

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(data_dir):
        if image_file.endswith('.png'):
            image_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(data_dir, image_file)
            results = model(image_path, save = True, project = output_dir, name='color', exist_ok=True)
            
            
    