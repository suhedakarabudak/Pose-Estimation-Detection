import cv2
import numpy as np
import time
import mediapipe as mp
import pandas as pd

# mpdrawing çizim noktlarını getirin değişken
# mp_pose pose modulünü içe aktarıyoruz

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Yeni video boyutları
new_width = 480
new_height = 480

# VIDEO FEED
cap = cv2.VideoCapture("./AiTrainer/wrongsquat2.mp4")

# Videoyu yeniden boyutlandırma fonksiyonu
def resize_frame(frame):
    return cv2.resize(frame, (new_width, new_height))

# Excel dosyası oluşturma ve sütun başlıklarını belirleme
columns = ["Frame No.", "Landmark ID", "Landmark Name", "X", "Y", "Z"]
data = []


# Landmark isimleri
landmark_names = ["Nose", "Left Eye Inner", "Left Eye", "Left Eye Outer", "Right Eye Inner", "Right Eye", "Right Eye Outer",
                  "Left Ear", "Right Ear", "Mouth Left", "Mouth Right", "Left Shoulder", "Right Shoulder", "Left Elbow",
                  "Right Elbow", "Left Wrist", "Right Wrist", "Left Pinky", "Right Pinky", "Left Index", "Right Index",
                  "Left Thumb", "Right Thumb", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle",
                  "Left Heel", "Right Heel", "Left Foot Index", "Right Foot Index"]

# mediapipe örneğini kurma
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    frame_no = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Yeni boyuta yeniden boyutlandır
        frame = resize_frame(frame)

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        #Eklem işaretlerinin çıkarıması
        try:
            landmarks = results.pose_landmarks.landmark
            for idx, landmark in enumerate(landmarks):
                data.append([frame_no, idx, landmark_names[idx], landmark.x, landmark.y, landmark.z])
        except:
            pass
        
        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        frame_no += 1

# DataFrame oluşturma
pred_df = pd.DataFrame(data, columns=columns)

excel_file_path = "pred_df.xlsx"  
# Excel dosyasına yazma
pred_df.to_excel(excel_file_path, index=False)


"""def merge_multiple_datasets(correct_squat_files, wrong_squat_files, merged_file):
    

    correct_squat_dataframes = []
    for file in correct_squat_files:
        correct_squat_dataframes.append(pd.read_excel(file))  # Excel dosyası okunurken pd.read_excel() kullanılmalı
    merged_correct_data = pd.concat(correct_squat_dataframes, ignore_index=True)
    merged_correct_data['Squat Type'] = 'Correct'
    

    wrong_squat_dataframes = []
    for file in wrong_squat_files:
        wrong_squat_dataframes.append(pd.read_excel(file))  # Excel dosyası okunurken pd.read_excel() kullanılmalı
    merged_wrong_data = pd.concat(wrong_squat_dataframes, ignore_index=True)
    merged_wrong_data['Squat Type'] = 'Wrong'
    

    merged_data = pd.concat([merged_correct_data, merged_wrong_data], ignore_index=True)
    
    # Birleştirilmiş veri setini dosyaya yaz
    merged_data.to_csv(merged_file, index=False)
    
    print("Tüm squat veri setleri başarıyla birleştirildi ve kaydedildi.")

# Kullanım örneği:
correct_squat_files = ["squat_landmarks.xlsx"]
wrong_squat_files = ["wrongsquat_landmarks.xlsx"]
merged_file = "merged_squat_data.csv"

merge_multiple_datasets(correct_squat_files, wrong_squat_files, merged_file)"""

