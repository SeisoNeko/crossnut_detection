from ultralytics import YOLO
import os

from utils.coco2yolo import convert_coco_to_yolo
import cv2

if __name__ == '__main__':

    dataset_path = 'dataset/v4'  

    train_label_path = os.path.join(dataset_path, 'labels/train')
    val_label_path = os.path.join(dataset_path, 'labels/val')
    test_label_path = os.path.join(dataset_path, 'labels/test')

    #將COCO格式的標註轉換為YOLO格式，執行一次就可以block了
    convert_coco_to_yolo(json_path = train_label_path, output_path = train_label_path)
    convert_coco_to_yolo(json_path = val_label_path, output_path = val_label_path)
    convert_coco_to_yolo(json_path = test_label_path, output_path = test_label_path)
    
    model = YOLO('yolov8n-seg.pt')
    image_folder = os.path.join(dataset_path, 'images/train')
    
    # Load segmentation labels
    segmentation_labels = {}
    for label_file in os.listdir(train_label_path):
        if label_file.endswith('.txt'):
            with open(os.path.join(train_label_path, label_file), 'r') as f:
                segmentation_labels[label_file] = f.readlines()
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.jpg')]

    '''
    # Display the first 5 images
    for image_file in image_files[:5]:  
        image = cv2.imread(image_file)
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'
    '''

    # Train the model
    results = model.train(data=os.path.join(dataset_path, 'roboflow_dataset.yaml'), epochs=100, imgsz=640, device='cuda')
    metrics = model.val()

    model.export(format='onnx')