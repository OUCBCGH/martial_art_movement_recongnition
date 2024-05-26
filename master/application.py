import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import mediapipe as mp
import json

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

# 加载训练好的动作识别模型
model = SETGCN(in_features=3, num_classes=5)  # 修改num_classes为实际的类别数
model.load_state_dict(torch.load('trained_model.pth'))
model.eval()

# 定义动作标签
#label_mapping = {0: '大', 1: '顺', 2: '拗',3:'小',4:'败'}  # 修改为实际的标签映射
label_mapping = {0: '0', 1: '1', 2: '2',3:'3',4:'4'}  # 修改为实际的标签映射

# 定义骨骼关键点检测模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 读取视频
url = 'http://192.168.47.203:81/stream'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 视频写出设置
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换颜色
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])

        keypoints = torch.tensor([keypoints], dtype=torch.float32)  # 转换为Tensor
        outputs = model(keypoints)
        _, predicted = torch.max(outputs.data, 1)
        predicted_label = label_mapping[predicted.item()]

        # 绘制关键点
        for landmark in results.pose_landmarks.landmark:
            cx, cy = int(landmark.x * frame_width), int(landmark.y * frame_height)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        # 显示预测结果
        cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    out.write(frame)
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
