from ultralytics import YOLO
from utils.transform import *

if __name__ == '__main__':
    
    """ model = YOLO('model/yolo11m.pt')
    yaml_path = r'dataset\original_v10\data.yaml'

    model.train(data=yaml_path, epochs=100, imgsz=640, device='cuda', batch=16, workers=8)
    
    # Validate the model
    metrics = model.val()  # no arguments needed, dataset and settings remembered
    metrics.box.map  # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps  # a list contains map50-95 of each category

    #save model
    model.export(format='onnx') """

    #inference
    model = YOLO('model/best.pt')
    path = r'temp\crops'
    model.predict(source=path, conf=0.5, save=True, project='output', name='test', exist_ok=True, retina_masks=True) #conf=0.5