import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.stdout.reconfigure(encoding='utf-8')

# 模型定义
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)  # Adding an additional GCN layer
        self.bn4 = BatchNorm(hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)  # Adding another additional GCN layer
        self.bn5 = BatchNorm(hidden_dim)
        self.dropout = nn.Dropout(p=0.5)  # Adjust Dropout rate to 0.5
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = self.bn5(x)
        x = torch.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# 参数
input_dim = 3  # 关键点维度
hidden_dim = 512
output_dim = 5  # 动作类别数量
model = GCN(input_dim, hidden_dim, output_dim)

# 加载 LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# 获取标签对应关系
label_mapping = dict(zip(range(len(label_encoder.classes_)), label_encoder.classes_))
print("Label Mapping:", label_mapping)

# 实时视频处理和预测
def process_video(video_path):
    # 加载模型
    model = GCN(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(r'C:\Users\Lenovo\martial_art_movement_recongnition\6m2d\gcn_model_0_8971.pth'))
    model.eval()

    # 初始化 Mediapipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 将 BGR 图像转换为 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 检测姿势
        results = pose.process(image)

        # 处理检测到的姿势
        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])

            keypoints = torch.tensor(keypoints, dtype=torch.float).view(-1, 3)

            # 创建图数据
            edge_index = []
            num_keypoints = len(keypoints)
            for i in range(num_keypoints - 1):
                edge_index.append([i, i + 1])
                edge_index.append([i + 1, i])
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            data = Data(x=keypoints, edge_index=edge_index_tensor)
            data.batch = torch.tensor([0] * num_keypoints, dtype=torch.long)  # 添加 batch 属性

            # 预测
            out = model(data)
            pred = out.argmax(dim=1)
            label = label_mapping[pred.item()]

            # 使用 Pillow 在图像上显示中文标签
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            font_path = "simhei.ttf"  # 请确保字体文件在指定路径
            font = ImageFont.truetype(font_path, 40)
            draw.text((50, 50), label, font=font, fill=(0, 255, 0))
            frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # 显示结果图像
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

# 处理视频文件
video_path = r'C:\Users\Lenovo\martial_art_movement_recongnition\6m2d\video\5.mp4'
process_video(video_path)
