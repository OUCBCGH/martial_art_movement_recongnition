import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import cv2
import numpy as np

#--------------------------load----------------------------------------------------
# 加载骨骼关键点数据
with open('keypoints_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

keypoints_data = np.array(data['keypoints'])
labels = np.array(data['labels'])

# 将标签转换为整数类型
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])

# 打印标签分布，确保正确加载
print(f"Loaded {len(keypoints_data)} keypoints with {len(labels)} labels.")
unique_labels, counts = np.unique(labels, return_counts=True)
print(f"Labels distribution: {dict(zip(unique_labels, counts))}")

#------------------------------train------------------------------------------
class SkeletonDataset(Dataset):
    def __init__(self, keypoints, labels):
        self.keypoints = keypoints
        self.labels = labels

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, idx):
        return torch.tensor(self.keypoints[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

class SETGCN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(SETGCN, self).__init__()
        self.gcn1 = nn.Conv1d(in_features, 64, kernel_size=1)
        self.gcn2 = nn.Conv1d(64, 128, kernel_size=1)
        self.gcn3 = nn.Conv1d(128, 256, kernel_size=1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch_size, num_joints, in_features) -> (batch_size, in_features, num_joints)
        x = self.gcn1(x)
        x = nn.ReLU()(x)
        x = self.gcn2(x)
        x = nn.ReLU()(x)
        x = self.gcn3(x)
        x = nn.ReLU()(x)
        x = torch.mean(x, dim=-1)
        x = self.fc(x)
        return x

# 数据准备
dataset = SkeletonDataset(keypoints_data, labels)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 模型初始化
model = SETGCN(in_features=3, num_classes=len(unique_labels))  # 使用唯一标签的数量作为类别数
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}')
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Model saved to 'trained_model.pth'")

train_model(model, dataloader, criterion, optimizer)
#----------------------evaluate------------------------------------
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')

# 假设有一个测试集
test_dataset = SkeletonDataset(keypoints_data, labels)  # 使用与训练集相同的数据进行示例
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)
evaluate_model(model, test_dataloader)
