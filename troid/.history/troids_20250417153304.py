import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Veriyi Yükle
df = pd.read_csv("Thyroid_Diff.csv")  # Dosya adını buraya yaz

# 2. Eksik verileri temizle
df = df.dropna()

# 3. Kategorik sütunları sayısal değerlere çevir
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

# 4. Giriş (X) ve hedef (y) verilerini ayır
X = df.drop("Recurred", axis=1)
y = df["Recurred"]

# 5. Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Ölçekleme (standardizasyon)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 7. Yapay Sinir Ağı (MLPClassifier) oluştur
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, activation='relu', solver='adam', random_state=42)

# 8. Modeli eğit
mlp.fit(X_train, y_train)

# 9. Tahmin yap
y_pred = mlp.predict(X_test)

# 10. Sonuçları değerlendir
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 11. Eğitim kaybı grafiği (varsa)
if hasattr(mlp, 'loss_curve_'):
    plt.plot(mlp.loss_curve_)
    plt.title("Eğitim Kaybı (Loss)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()
