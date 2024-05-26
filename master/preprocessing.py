import json
import cv2
import numpy as np
import mediapipe as mp
import os

def process_video(video_file, annotation_file):
    # 加载VIA标注文件
    with open(annotation_file, 'r', encoding='utf-8') as f:
        via_data = json.load(f)

    # 提取标注信息
    annotations = via_data['metadata']

    # 转换为训练数据格式并检查段数据格式
    training_data = []
    for annotation in annotations.values():
        segment = annotation['z']
        label = annotation['av']['1']
        if len(segment) == 2:  # 确保段数据有开始和结束时间
            training_data.append((segment, label))

    print(f"Parsed {len(training_data)} annotations for {video_file}.")

    # 提取关键帧函数，并保留标签
    def extract_keyframes_with_labels(video_path, segments_labels):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return [], []

        keyframes = []
        labels = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Video FPS: {fps}, Total Frames: {total_frames}")

        for segment, label in segments_labels:
            start_time, end_time = segment
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)

            if start_frame >= total_frames or end_frame >= total_frames:
                print(f"Warning: Segment {segment} is out of video bounds.")
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            while cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
                success, frame = cap.read()
                if success:
                    keyframes.append(frame)
                    labels.append(label)
                else:
                    break

        cap.release()
        return keyframes, labels

    # 提取关键帧并进行去噪处理
    keyframes, labels = extract_keyframes_with_labels(video_file, training_data)

    if keyframes:
        def bilateral_filter(frames):
            filtered_frames = []
            for frame in frames:
                filtered_frame = cv2.bilateralFilter(frame, 9, 75, 75)
                filtered_frames.append(filtered_frame)
            return filtered_frames

        filtered_keyframes = bilateral_filter(keyframes)
        print(f"Filtered {len(filtered_keyframes)} keyframes for {video_file}.")
    else:
        print(f"No keyframes extracted for {video_file}.")
        return [], []

    # 骨骼关键点检测
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

    keypoints_data = []
    filtered_labels = []

    for frame, label in zip(filtered_keyframes, labels):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z])
            keypoints_data.append(keypoints)
            filtered_labels.append(label)

    print(f"Extracted keypoints from {len(keypoints_data)} frames for {video_file}.")
    return keypoints_data, filtered_labels

# 视频和标注文件列表
video_annotation_pairs = [
    ('vedio/1.mp4', 'JSON/1.json'),
    ('vedio/2.mp4', 'JSON/2.json'),
    ('vedio/3.mp4', 'JSON/3.json'),
    ('vedio/4.mp4', 'JSON/4.json'),
    ('vedio/5.mp4', 'JSON/5.json')
    # 添加更多的 (视频文件, 标注文件) 对
]

all_keypoints_data = []
all_labels = []

# 处理每个视频和对应的标注文件
for video_file, annotation_file in video_annotation_pairs:
    keypoints_data, labels = process_video(video_file, annotation_file)
    all_keypoints_data.extend(keypoints_data)
    all_labels.extend(labels)

# 保存所有视频的骨骼关键点数据
with open('keypoints_data.json', 'w', encoding='utf-8') as f:
    json.dump({'keypoints': all_keypoints_data, 'labels': all_labels}, f, ensure_ascii=False, indent=4)

print(f"Combined keypoints data from {len(video_annotation_pairs)} videos saved to 'keypoints_data.json'.")
