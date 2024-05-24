import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

standard_data = {
    "y1": 90, "y2": 90, "y3": 90, "y4": 90, "y5": 90, "y6": 90, 
    "y7": 90, "y8": 90, "y9": 90, "y10": 90, "y11": 90, "y12": 90, "y13": 90
}

weights = {
    "z1": 1, "z2": 1, "z3": 1, "z4": 1, "z5": 1, "z6": 1, 
    "z7": 1, "z8": 1, "z9": 1, "z10": 1, "z11": 1, "z12": 1, "z13": 1
}

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return angle if angle <= 180.0 else 360.0 - angle

def calculate_score(angles, standard_data, weights):
    score = 100
    for i, angle in enumerate(angles):
        angle_name = f'y{i+1}'
        if angle_name in standard_data:
            score -= weights[f'z{i+1}'] * np.abs(angle - standard_data[angle_name])
    return score

def main(url):
    
    cap = cv2.VideoCapture(url)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                # 计算角度
                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
                left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                left_wrist_angle = calculate_angle(left_elbow, left_wrist, [left_wrist[0], left_wrist[1] + 1])
                right_wrist_angle = calculate_angle(right_elbow, right_wrist, [right_wrist[0], right_wrist[1] + 1])
                hip_split_angle = calculate_angle(left_hip, [0.5 * (left_hip[0] + right_hip[0]), 0.5 * (left_hip[1] + right_hip[1])], right_hip)
                left_leg_angle = calculate_angle(left_knee, left_hip, [left_hip[0] + 1, left_hip[1]])
                right_leg_angle = calculate_angle(right_knee, right_hip, [right_hip[0] + 1, right_hip[1]])

                angles = [
                    left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle, 
                    left_hip_angle, right_hip_angle, left_shoulder_angle, right_shoulder_angle, 
                    left_wrist_angle, right_wrist_angle, hip_split_angle, left_leg_angle, right_leg_angle
                ]

                # 打印角度
                print(f"Angles: {angles}")

                # 计算得分
                score = calculate_score(angles, standard_data, weights)
                print(f"Score: {score}")

                # 在图像上显示得分
                cv2.putText(image, f'Score: {score}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                # 绘制姿势
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow('Mediapipe Pose', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    url = 'http://192.168.164.203:81/stream'
    main(url)