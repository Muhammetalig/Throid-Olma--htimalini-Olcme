import pandas as pd
import numpy as np

# --------------------------- Yardımcı Fonksiyonlar ----------------------------

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def one_hot_encode(column):
    return pd.get_dummies(column, prefix=column.name)

def normalize(df):
    return (df - df.min()) / (df.max() - df.min())

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# ------------------------------ Veri Yükleme ---------------------------------

df = pd.read_csv("thyroid_diff.csv")

# ---------------------- Kategorik Sütunları Kodlama --------------------------

categorical_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function',
                    'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 'Risk',
                    'T', 'N', 'M', 'Stage', 'Response']

df_encoded = df.copy()
for col in categorical_cols:
    df_encoded = df_encoded.drop(columns=[col]).join(one_hot_encode(df[col]))

# ----------------------------- Özellikler & Hedef ----------------------------

X = df_encoded.drop(columns=["Recurred"])
X = normalize(X)
y = df["Recurred"].values.reshape(-1, 1)

# ----------------------------- Eğitim & Test Bölme ---------------------------

split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train = X.iloc[:split_index].values
y_train = y[:split_index]
X_test = X.iloc[split_index:].values
y_test = y[split_index:]

# ----------------------------- Ağırlık Başlatma ------------------------------

input_size = X_train.shape[1]
hidden_size = 20
output_size = 1
np.random.seed(42)

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# ----------------------------- Eğitim Döngüsü --------------------------------

epochs = 1000
lr = 0.1

for epoch in range(epochs):
    # Forward propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Kayıp
    loss = binary_cross_entropy(y_train, A2)

    # Geri yayılım
    dA2 = A2 - y_train
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Ağırlık güncelleme
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# ---------------------------- Test Değerlendirme -----------------------------

Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

predictions = (A2_test > 0.5).astype(int)
accuracy = np.mean(predictions == y_test)
print(f"\nTest Doğruluğu: %{accuracy * 100:.2f}")
