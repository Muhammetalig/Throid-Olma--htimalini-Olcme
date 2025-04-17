import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------- 1. VERİ YÜKLE -------------------
try:
    data = pd.read_csv('thyroid_diff.csv')
except FileNotFoundError:
    print("Hata: 'thyroid_diff.csv' dosyası bulunamadı.")
    exit()

# ------------------- 2. LABEL ENCODING -------------------
from collections import defaultdict

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

# Normalizasyon (0-1 arası)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# ------------------- 4. VERİYİ BÖL (80-20) -------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ------------------- 5. AKTİVASYON FONKSİYONLARI -------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ------------------- 6. MODEL PARAMETRELERİ -------------------
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1
epochs = 500
learning_rate = 0.1

# ------------------- 7. AĞ AĞIRLIKLARI -------------------
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

losses = []

# ------------------- 8. EĞİTİM -------------------
for epoch in range(epochs):
    # FORWARD
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # KAYIP
    loss = np.mean((y_train - A2) ** 2)
    losses.append(loss)

    # BACKPROP
    dA2 = -(y_train - A2)
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # GÜNCELLEME
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 50 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# ------------------- 9. TEST -------------------
Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)
Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

predictions = (A2_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"\nTest Doğruluğu: {accuracy * 100:.2f}%")

# ------------------- 10. LOSS GRAFİĞİ -------------------
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Sürecindeki Loss Grafiği")
plt.grid(True)
plt.legend()
plt.show()

# ------------------- 11. KULLANICI VERİSİYLE TAHMİN -------------------
def get_user_input(label_encoders):
    print("\n📋 Yeni bir kişi için bilgi girin:")
    person = []

    for col in data.columns:
        if col == 'Recurred':
            continue
        if col in label_encoders:
            options = list(label_encoders[col].keys())
            print(f"{col} seçenekleri: {options}")
            val = input(f"{col}: ")
            while val not in label_encoders[col]:
                val = input(f"Lütfen geçerli bir değer girin ({options}): ")
            person.append(label_encoders[col][val])
        else:
            val = input(f"{col} (sayı): ")
            person.append(float(val))
    
    person = np.array(person).reshape(1, -1)
    # normalize et
    person = (person - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

    # tahmin et
    z1 = np.dot(person, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    print(f"\n📈 Yineleme Olasılığı: {a2[0][0]*100:.2f}%")
    print("📌 Sonuç: ", "Yineleme Var" if a2[0][0] > 0.5 else "Yineleme Yok")

# Kullanıcıdan bilgi al ve tahmin yap
get_user_input(label_encoders)
