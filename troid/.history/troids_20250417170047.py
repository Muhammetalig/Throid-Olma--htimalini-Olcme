import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# ------------------- 1. VERİ YÜKLE -------------------
try:
    data = pd.read_csv('thyroid_diff.csv')
except FileNotFoundError:
    print("Hata: 'thyroid_diff.csv' dosyası bulunamadı.")
    exit()

# ------------------- 2. LABEL ENCODING -------------------
def label_encode_column(column):
    unique_values = sorted(column.unique())
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return column.map(mapping), mapping

label_encoders = defaultdict(dict)
for col in data.columns:
    if data[col].dtype == 'object':
        data[col], mapping = label_encode_column(data[col])
        label_encoders[col] = mapping

# ------------------- 3. ÖZELLİK-HEDEF AYRIMI -------------------
X = data.drop(columns=['Recurred']).values
y = data['Recurred'].values.reshape(-1, 1)

# ------------------- 4. VERİYİ NORMALİZE ET -------------------
# Normalizasyon yerine veriyi manuel olarak standardize ediyoruz:
# (her sütun için: ortalama 0, standart sapma 1 yapıyoruz)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# ------------------- 5. VERİYİ BÖL (80-20) -------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------- 6. AKTİVASYON FONKSİYONLARI -------------------
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ------------------- 7. MODEL PARAMETRELERİ -------------------
input_size = X_train.shape[1]
hidden_size1 = 20  # Gizli katman boyutunu artırıyoruz
hidden_size2 = 10  # İkinci bir gizli katman ekliyoruz
output_size = 1
epochs = 1000  # Epoch sayısını artırıyoruz
learning_rate = 0.01  # Öğrenme oranını düşürüyoruz

# ------------------- 8. AĞ AĞIRLIKLARI -------------------
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size1)
b1 = np.zeros((1, hidden_size1))
W2 = np.random.randn(hidden_size1, hidden_size2)
b2 = np.zeros((1, hidden_size2))
W3 = np.random.randn(hidden_size2, output_size)
b3 = np.zeros((1, output_size))

losses = []

# ------------------- 9. EĞİTİM -------------------
for epoch in range(epochs):
    # FORWARD
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = relu(Z2)
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    # KAYIP (Binary Cross-Entropy)
    loss = -np.mean(y_train * np.log(A3) + (1 - y_train) * np.log(1 - A3))
    losses.append(loss)

    # BACKPROP
    dA3 = -(y_train - A3)
    dZ3 = dA3 * sigmoid_derivative(A3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * relu_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # GÜNCELLEME
    W3 -= learning_rate * dW3
    b3 -= learning_rate * db3
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# ------------------- 10. TEST -------------------
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = relu(Z2_test)
Z3_test = np.dot(A2_test, W3) + b3
A3_test = sigmoid(Z3_test)

predictions = (A3_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"\nTest Doğruluğu: {accuracy * 100:.2f}%")

# ------------------- 11. LOSS GRAFİĞİ -------------------
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Sürecindeki Loss Grafiği")
plt.grid(True)
plt.legend()
plt.show()
