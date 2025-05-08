import os
import cv2
import yaml
import random
import albumentations as A
from tqdm import tqdm

def load_yolo_data(yaml_path):
    """載入 YOLO 格式的 YAML 檔案，獲取圖片和標籤的相對路徑。"""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    train_images_rel_path = data.get('train')
    val_images_rel_path = data.get('val')
    yaml_dir = os.path.dirname(yaml_path) # 獲取 YAML 檔案所在的目錄
    print(f"YAML 檔案目錄：{yaml_dir}")
    print(f"訓練集圖片相對路徑：{train_images_rel_path}, 驗證集圖片相對路徑：{val_images_rel_path}")
    return train_images_rel_path, val_images_rel_path, yaml_dir

def load_yolo_labels(label_path):
    """載入 YOLO 格式的標籤檔案。"""
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    labels.append([x_center, y_center, width, height, class_id])
    except FileNotFoundError:
        print(f"警告：標籤檔案 {label_path} 未找到。")
    return labels

def save_yolo_labels(label_path, labels):
    """保存 YOLO 格式的標籤檔案。"""
    with open(label_path, 'w') as f:
        for label in labels:
            x_center, y_center, width, height, class_id = label
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def apply_augmentations(image, bboxes, augmentations):
    """應用 albumentations 定義的增強。"""
    augmented = augmentations(image=image, bboxes=bboxes)
    return augmented['image'], augmented['bboxes']

def process_images(images_rel_dir, output_images_dir, output_labels_dir, augment, yaml_dir):
    """處理指定相對目錄下的所有圖片和標籤。"""
    images_abs_dir = os.path.abspath(os.path.join(yaml_dir, images_rel_dir))
    labels_abs_dir = os.path.abspath(os.path.join(yaml_dir, images_rel_dir.replace('images', 'labels'))) # 假設 labels 在同級目錄的 labels 子目錄下

    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_abs_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in tqdm(image_files, desc=f"Processing {os.path.basename(images_abs_dir)}"):
        image_path = os.path.join(images_abs_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_abs_dir, label_file)

        image = cv2.imread(image_path)
        if image is None:
            print(f"警告：無法讀取圖片 {image_path}。")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        labels = load_yolo_labels(label_path)
        bboxes = [([l[0] - l[2] / 2, l[1] - l[3] / 2, l[0] + l[2] / 2, l[1] + l[3] / 2], l[4]) for l in labels]

        augmented_image = image
        augmented_bboxes_with_class = []

        if augment:
            augmented = augment(image=image, bboxes=[bbox[:4] for bbox in bboxes], class_labels=[bbox[4] for bbox in bboxes])
            augmented_image = augmented['image']
            for bbox, class_id in zip(augmented['bboxes'], augmented['class_labels']):
                augmented_bboxes_with_class.append(list(bbox) + [class_id])

            augmented_labels_yolo = []
            aug_h, aug_w = augmented_image.shape[:2]
            for bbox in augmented_bboxes_with_class:
                x_min, y_min, x_max, y_max, class_id = bbox
                x_center = (x_min + x_max) / 2 / aug_w
                y_center = (y_min + y_max) / 2 / aug_h
                width = (x_max - x_min) / aug_w
                height = (y_max - y_min) / aug_h
                augmented_labels_yolo.append([x_center, y_center, width, height, class_id])
        else:
            augmented_labels_yolo = [[l[0], l[1], l[2], l[3], l[4]] for l in labels]
            augmented_image = image

        output_image_path = os.path.join(output_images_dir, image_file)
        cv2.imwrite(output_image_path, cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR))

        output_label_path = os.path.join(output_labels_dir, label_file)
        save_yolo_labels(output_label_path, augmented_labels_yolo)

if __name__ == "__main__":
    yaml_file = 'dataset/original_v10/data.yaml' # 替換為您的 YAML 檔案路徑
    output_root_dir = 'augmented_dataset' # 輸出增強後資料集的根目錄

    train_images_rel_path, val_images_rel_path, yaml_dir = load_yolo_data(yaml_file)

    output_train_images_dir = os.path.join(output_root_dir, 'train/images')
    output_train_labels_dir = os.path.join(output_root_dir, 'train/labels')
    output_val_images_dir = os.path.join(output_root_dir, 'val/images')
    output_val_labels_dir = os.path.join(output_root_dir, 'val/labels')

    augmentations = A.Compose([
        A.Rotate(limit=(-5, 5), p=0.5),
        A.Affine(shear=(-5, 5), p=0.5),
        A.Perspective(scale=(0.01, 0.05), p=0.5),
        A.RandomScale(scale_limit=(0.8, 1.2), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.Blur(blur_limit=3, p=0.3),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    # 處理訓練集
    process_images(
        train_images_rel_path,
        output_train_images_dir,
        output_train_labels_dir,
        augment=lambda image=None, bboxes=None, class_labels=None: augmentations(image=image, bboxes=bboxes, class_labels=class_labels) if image is not None else None,
        yaml_dir=yaml_dir
    )

    # 處理驗證集
    process_images(
        val_images_rel_path,
        output_val_images_dir,
        output_val_labels_dir,
        augment=None,
        yaml_dir=yaml_dir
    )

    print(f"增強後的資料集已保存到 {output_root_dir} 目錄。")