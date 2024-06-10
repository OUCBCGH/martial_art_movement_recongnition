# -----------导入库-----------
import json
import cv2
import numpy as np
import mediapipe as mp
from moviepy.editor import VideoFileClip
import os

# 初始化 MediaPipe Pose 模型
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

def load_data(json_path, video_path):
    print(f"Loading data from {json_path} and {video_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    video = VideoFileClip(video_path)
    return annotations, video

def extract_keypoints_from_frame(frame):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    keypoints = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            keypoints.append([landmark.x, landmark.y, landmark.z])
    return keypoints

def extract_keypoints_use_json(json_path, video_path):
    keypoints_data = []
    annotations, video = load_data(json_path, video_path)
    metadata = annotations['metadata']
    
    for metadata_id, segment in metadata.items():
        start_time, end_time = segment['z']
        label = segment['av']['1']
        for t in np.arange(start_time, end_time, 1 / video.fps):
            frame = video.get_frame(t)
            keypoints = extract_keypoints_from_frame(frame)
            if keypoints:
                keypoints_data.append({
                    'label': label,
                    'keypoints': keypoints
                })
    print(f"Extracted {len(keypoints_data)} keypoints from {video_path}")
    return keypoints_data

def save_all_keypoints(output_file, all_keypoints):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_keypoints, f, ensure_ascii=False, indent=4)
    print(f"All keypoints data saved to '{output_file}'")

def get_file_pairs(video_dir, json_dir):
    video_files = {os.path.splitext(f)[0]: f for f in os.listdir(video_dir) if f.endswith('.mp4')}
    json_files = {os.path.splitext(f)[0]: f for f in os.listdir(json_dir) if f.endswith('.json')}
    common_files = video_files.keys() & json_files.keys()
    file_pairs = [(os.path.join(video_dir, video_files[name]), os.path.join(json_dir, json_files[name])) for name in common_files]
    return file_pairs

# 使用区
##文件路径定义
video_annotation_pairs = video_annotation_pairs = [
    #('output_video/1_5fps.mp4', 'JSON/1.json'),
    ('output_video/2_5fps.mp4', 'JSON/2.json'),
    ('output_video/3_5fps.mp4', 'JSON/3.json'),
    ('output_video/4_5fps.mp4', 'JSON/4.json'),
    #('output_video/5_5fps.mp4', 'JSON/5.json'),
    ('output_video/6_5fps.mp4', 'JSON/5.json'),
    ('output_video/7_5fps.mp4', 'JSON/5.json'),
    ('output_video/53101_5fps.mp4', 'JSON/53101.json'),
    ('output_video/53102_5fps.mp4', 'JSON/53102.json'),
    ('output_video/53103_5fps.mp4', 'JSON/53103.json'),
    ('output_video/53104_5fps.mp4', 'JSON/53104.json'),
    ('output_video/60701_5fps.mp4', 'JSON/60701.json'),
    ('output_video/60702_5fps.mp4', 'JSON/60702.json'),
    ('output_video/60703_5fps.mp4', 'JSON/60703.json'),
    ('output_video/60704_5fps.mp4', 'JSON/60704.json'),
    ('output_video/60705_5fps.mp4', 'JSON/60705.json'),
    ('output_video/60706_5fps.mp4', 'JSON/60706.json'),
    ('output_video/60707_5fps.mp4', 'JSON/60707.json'),
    ('output_video/60708_5fps.mp4', 'JSON/60708.json'),
    ('output_video/60709_5fps.mp4', 'JSON/60709.json'),
    ('output_video/60710_5fps.mp4', 'JSON/60710.json'),
    ('output_video/60711_5fps.mp4', 'JSON/60711.json'),
    ('output_video/60712_5fps.mp4', 'JSON/60712.json'),
    #('output_video/60801_5fps.mp4', 'JSON/60801.json'),
    ('output_video/60802_5fps.mp4', 'JSON/60802.json'),
    #('output_video/60803_5fps.mp4', 'JSON/60803.json'),
    ('output_video/60804_5fps.mp4', 'JSON/60804.json'),
    ('output_video/60805_5fps.mp4', 'JSON/60805.json'),
    ('output_video/60806_5fps.mp4', 'JSON/60806.json'),
    ('output_video/60807_5fps.mp4', 'JSON/60807.json')

    # (视频文件, 标注文件)
]
# ##文件路径匹配
# video_dir = 'output_video'
# json_dir = 'JSON'

# video_annotation_pairs = get_file_pairs(video_dir, json_dir)
# print(f"Found {len(video_annotation_pairs)} pairs of video and JSON files")

all_keypoints = []

for video_path, json_path in video_annotation_pairs:
    try:
        keypoints_data = extract_keypoints_use_json(json_path, video_path)
        all_keypoints.extend(keypoints_data)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

output_file = 'all_keypoints.json'
save_all_keypoints(output_file, all_keypoints)

print(f"Total extracted keypoints: {len(all_keypoints)}")
