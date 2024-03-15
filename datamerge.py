import pandas as pd
import os


def merge_multiple_datasets(correct_squat_files, wrong_squat_files, merged_file):
    

    correct_squat_dataframes = []
    for file in correct_squat_files:
        correct_squat_dataframes.append(pd.read_excel(file))
    merged_correct_data = pd.concat(correct_squat_dataframes, ignore_index=True)
    merged_correct_data['Squat Type'] = 'Correct'
    

    wrong_squat_dataframes = []
    for file in wrong_squat_files:
        wrong_squat_dataframes.append(pd.read_excel(file))
    merged_wrong_data = pd.concat(wrong_squat_dataframes, ignore_index=True)
    merged_wrong_data['Squat Type'] = 'Wrong'
    

    merged_data = pd.concat([merged_correct_data, merged_wrong_data], ignore_index=True)
    
    # Birleştirilmiş veri setini dosyaya yaz
    merged_data.to_csv(merged_file, index=False)
    
    print("Tüm squat veri setleri başarıyla birleştirildi ve kaydedildi.")

# Kullanım örneği:
correct_squat_files = ["squat_landmarks.xlsx", "yenisquat.xlsx"]
wrong_squat_files = ["wrongsquat_landmarks.xlsx", "yenisquatyanlis.xlsx"]
merged_file = "new_merged_squat_data.csv"

merge_multiple_datasets(correct_squat_files, wrong_squat_files, merged_file)
