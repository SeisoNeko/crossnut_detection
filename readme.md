# 道路挖掘偵測專案 (Road Excavation Detection)

## 專案概述
這個專案使用 YOLOv8 進行道路挖掘工程的語義分割 (semantic segmentation) 檢測。系統可以識別和標記出道路上的挖掘區域，支援影像分析與處理。

## 功能
- 使用 YOLOv8 進行道路挖掘區域的語意分割
- 支援 COCO 格式資料集轉換為 YOLO 訓練格式
- 提供預訓練模型用於直接推論
- 自動化資料增強和模型訓練流程

## 環境需求
- Python 3.x
- PyTorch
- Ultralytics
- OpenCV
- NumPy
- PIL (Python Imaging Library)

## 檔案說明
- `main.py` - 使用預訓練模型進行影像推論
- `yolo_train.py` - YOLOv8 模型訓練與驗證
- `utils.py` - 實用工具函數，包含 COCO 到 YOLO 格式轉換
- `adjust_ann.py` - 資料集標註調整工具(目前尚無用)
- `cross_detect.py` - 交叉檢測功能 (已棄用)

## 資料夾結構
- `data/` - 原始影像資料
- `dataset/v4/` - 處理過的訓練資料集
- `model/` - 儲存預訓練模型
- `output/` - 預測結果輸出資料夾
- `runs/` - 訓練過程記錄

## 使用說明

### 模型訓練
```bash
python yolo_train.py
```

### 執行預測
```bash
python main.py
```

### 標註格式轉換
使用 `utils.py` 中的 `convert_coco_to_yolo` 函數將 COCO 格式轉換為 YOLO 格式：
```python
from utils import convert_coco_to_yolo
convert_coco_to_yolo(json_path='path/to/coco', output_path='path/to/output')
```

## 模型資訊
- 基礎模型: YOLOv8n-seg
- 訓練資料集: 道路挖掘區域影像
- 訓練參數: 100個 epochs，640x640 影像大小

## 注意事項
- 確保 `model/best.pt` 檔案存在於運行 `main.py` 之前
- 資料集結構需符合 YOLOv8 要求格式