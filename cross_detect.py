import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
import json
import glob
import os
import numpy as np
import random
from PIL import Image

class RulerIntersectionDataset(Dataset):
    def __init__(self, image_folder, annotation_file, transform=None):
        self.image_paths = glob.glob(os.path.join(image_folder, "*.jpg"))  # 讀取所有圖片
        self.transform = transform
        
        # 讀取標註數據
        with open(annotation_file, "r") as f:
            self.labels = json.load(f)

        # 確保只載入有標註的圖片
        self.image_paths = [img for img in self.image_paths if os.path.basename(img) in self.labels]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 轉換為 RGB 格式
        
        # 取得對應的標註
        filename = os.path.basename(img_path)
        label = self.labels[filename]  # 取得交點座標
        x, y = label["x"], label["y"]
        
        h, w = image.shape[:2]  # 獲取原始圖片尺寸
        target = torch.tensor([x, y], dtype=torch.float32)  # 轉換為 Tensor

        # 影像增強（動態調整標註）
        if self.transform:
            image, target = self.apply_transform(image, target, w, h)

        return image, target

    def apply_transform(self, image, target, w, h):
        """
        這個函數確保標註 (x, y) 位置會根據變換方式動態調整
        """
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 先縮放，等比例變換 x, y
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

        # **水平方向翻轉**
        if torch.rand(1) < 0.5:  # 50% 機率翻轉
            image = np.fliplr(image).copy()  # 翻轉影像
            target[0] = w - target[0]  # x 軸鏡像

        # **隨機旋轉**
        angle = np.random.uniform(-10, 10)  # -10° ~ 10°
        if abs(angle) > 0:
            image = Image.fromarray(image)  # 轉換為 PIL Image
            image = image.rotate(angle, expand=True)  # 旋轉圖片
            cx, cy = w / 2, h / 2  # 計算旋轉中心
            x, y = target.numpy()

            # 計算旋轉後的新座標
            angle_rad = np.radians(angle)
            x_new = int((x - cx) * np.cos(angle_rad) - (y - cy) * np.sin(angle_rad) + cx)
            y_new = int((x - cx) * np.sin(angle_rad) + (y - cy) * np.cos(angle_rad) + cy)

            target = torch.tensor([x_new, y_new], dtype=torch.float32)  # 更新座標

        #將image變回numpy.ndarray
        image = np.array(image)

        # **應用所有變換**
        transform_pipeline = transforms.Compose(transform_list)
        image = transform_pipeline(image)

        # **縮放標註 (x, y)**
        target[0] = target[0] * (224 / w)  # x 按比例縮放
        target[1] = target[1] * (224 / h)  # y 按比例縮放

        return image, target


# 讀取完整 Dataset（不指定 transform）
dataset = RulerIntersectionDataset(image_folder="data/images", annotation_file="data/annotations.json")

# 切分 Train / Validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 確保 Train & Val 使用不同的 Transform
train_dataset.dataset.transform = True  # 啟用數據增強
val_dataset.dataset.transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"訓練集: {len(train_dataset)} 張, 驗證集: {len(val_dataset)} 張")
# 使用 ResNet18 作為特徵提取
class RulerIntersectionModel(nn.Module):
    def __init__(self):
        super(RulerIntersectionModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 2)  # 只輸出 (x, y) 座標

        #初始化最後一層
        nn.init.xavier_uniform_(self.resnet.fc.weight)
        nn.init.zeros_(self.resnet.fc.bias)

    def forward(self, x):
        return self.resnet(x)

# 設定模型、損失函數與優化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RulerIntersectionModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 訓練模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if torch.isnan(loss):
                    print("Warning: NaN detected in loss!")
                    continue

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 計算驗證集上的損失
    val_loss = 0.0
    for images, labels in val_loader:
        model.eval()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")

#隨機挑幾張圖片測試模型
model.eval()
for i in range(5):
    idx = random.randint(0, len(val_dataset))
    image, target = val_dataset[idx]
    image = image.unsqueeze(0).to(device)
    output = model(image)
    output = output.squeeze().detach().cpu().numpy()
    print(f"真實座標: {target.numpy()}, 預測座標: {output}")


# 儲存模型
torch.save(model.state_dict(), "ruler_intersection_model.pth")
