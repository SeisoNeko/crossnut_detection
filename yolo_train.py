from ultralytics import YOLO
import os

def coco2yolo(dataset_path):
    train_path = os.path.join(dataset_path, 'train')
    val_path = os.path.join(dataset_path, 'val')
    test_path = os.path.join(dataset_path, 'test')

if __name__ == '__main__':

    dataset_path = 'dataset/v4'  
    coco2yolo(dataset_path)

    model = YOLO('yolov8n-seg.pt')  
    results = model.train(data=os.path.join(dataset_path, 'roboflow_dataset.yaml'), epochs=100, imgsz=640, device='cuda')
    metrics = model.val()

    model.export(format='onnx')