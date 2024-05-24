import cv2
import mediapipe as mp

# 初始化 MediaPipe Pose 模块
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def main(video_source=0):
    # 打开视频源，可以是摄像头或视频文件
    cap = cv2.VideoCapture(video_source)

    # 使用 mp_pose.Pose 进行姿势检测
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 将图像从 BGR 转换为 RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # 处理图像并获取姿势结果
            results = pose.process(image)

            # 将图像转换回 BGR，以便 OpenCV 进行处理
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 绘制姿势关键点和连接线
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            # 显示结果图像
            cv2.imshow('Mediapipe Pose', image)

            # 按 'q' 键退出循环
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 传入视频文件路径或摄像头索引（默认0，即第一个摄像头）
    main("梅花拳示例.mp4")