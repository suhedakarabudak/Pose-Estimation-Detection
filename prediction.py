import pandas as pd
import joblib
import numpy as np

# Videodan alınan verileri yükleyin (uygun bir şekilde değiştirin)
pred_df = pd.read_excel("pred_df.xlsx")

# Modeli yükleyin
model = joblib.load("model1.pkl")

# Modelin eğitildiği ön işleme adımlarını yeniden uygulayın
# Ön işleme adımlarınıza göre videodan alınan verileri düzenleyin
# Veri setini yükle

# Veri ön işleme
def preprocess_data(df):
    # İstenmeyen öğeleri kaldırın
    unwanted_items = ['Inner', 'Outer','Nose','Shoulder','Ear','Mouth','Pinky ', 'Thumb','Eye']
    new_df = df[~df['Landmark Name'].str.contains('|'.join(unwanted_items))]
    
    # Gereksiz sütunları kaldırın
    columns_drop = ['Landmark ID', 'Frame No.']
    new_df = new_df.drop(columns=columns_drop)
    
    return new_df



# Özellik çıkarımı fonksiyonu
def extract_features_from_row(row):
    x = row['X']
    y = row['Y']
    z = row['Z']
    return [x, y, z]

preprocessed_data = preprocess_data(pred_df)

X_new = np.array([extract_features_from_row(row) for _, row in preprocessed_data.iterrows()])

# Tahmin yapın
predictions = model.predict(X_new)  # Videodan gelen verileri modelde kullanarak tahmin yapın
# Tahminleri yazdırın

print("Predictions:", predictions)
