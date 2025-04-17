import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 📥 1. VERİYİ YÜKLE
df = pd.read_csv("Thyroid_Diff.csv")  # Dosya adını doğru yaz

# 🔍 2. EKSİK VERİLERİ TEMİZLE
df = df.dropna()

# 🔄 3. KATEGORİK DEĞERLERİ SAYISAL YAP
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# 🎯 4. GİRİŞ (X) ve HEDEF (y) VERİLERİNİ AYIR
X = df.drop("Recurred", axis=1)  # hedef değişken
y = df["Recurred"]

# ✂️ 5. EĞİTİM VE TEST VERİSİNE BÖL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📏 6. ÖLÇEKLEME
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🧠 7. YSA MODELİNİ OLUŞTUR
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Binary classification için

# ⚙️ 8. DERLEME
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 📈 9. MODELİ EĞİT
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2)

# 🧪 10. MODELİ DEĞERLENDİR
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Doğruluğu: %{accuracy*100:.2f}")

# 🔮 11. TAHMİN YAP
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# 📊 12. SONUÇLARI GÖSTER
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

# 📉 13. EĞİTİM GRAFİĞİ
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title("Model Doğruluğu")
plt.xlabel("Epoch")
plt.ylabel("Doğruluk")
plt.legend()
plt.grid(True)
plt.show()
