from ultralytics import YOLO
import os

if __name__ == '__main__':
    
    model = YOLO('yolov8n-seg.pt')
    dataset_path = 'dataset/colored'

    #將除了最後的全連接層以外的所有層都設置為不訓練
    for param in model.parameters():
        param.requires_grad = False
    # 設置最後的全連接層為可訓練
    for param in model.model.model[-1].parameters():
        param.requires_grad = True

    model.train(data=os.path.join(dataset_path, 'data.yaml'), epochs=100, imgsz=640, device='cuda')
    metrics = model.val()

    model.export(format='onnx')