import ultralytics
import os

if __name__ == '__main__':
    model = ultralytics.YOLO('model/best.pt')
    data_dir = 'dataset/v4/images/test'

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in os.listdir(data_dir):
        if image_file.endswith('.jpg'):
            image_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(data_dir, image_file)
            results = model(image_path, save = True, project = output_dir, name='segmentation', exist_ok=True)
            
            
    