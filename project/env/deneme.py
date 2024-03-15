import cv2
import numpy as np
import mediapipe as mp

# Mediapipe Pose çözümünü ve çizim araçlarını tanımlama
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Squat durumunu belirlemek için fonksiyon
def determine_squat_status(left_angle, right_angle):
    if left_angle < 32 and right_angle < 32:
        return "s1"
    elif 35 <= left_angle <= 65 and 35 <= right_angle <= 65:
        return "s2"
    elif 75 <= left_angle <= 95 and 75 <= right_angle <= 95:
        return "s3"
    else:
        return "Unknown"

# İki vektör arasındaki açıyı hesaplamak için fonksiyon
def calculate_angle(a, b, c):
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Video dosyasını açma
cap = cv2.VideoCapture("./AiTrainer/squat.mp4")

# Mediapipe Pose ile işlem yapma
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

        try:
            landmarks = results.pose_landmarks.landmark
            left_angle = calculate_angle((landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y),
                                         (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y),
                                         (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y))
            right_angle = calculate_angle((landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y),
                                          (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y),
                                          (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y))
            
            # Squat durumunu belirleme
            squat_status = determine_squat_status(left_angle, right_angle)
            cv2.putText(image, f"Squat Status: {squat_status}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # İskelet çizgilerini ekleme
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))
        except:
            pass

        cv2.imshow('Squat Analysis', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

