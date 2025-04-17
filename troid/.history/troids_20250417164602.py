import pandas as pd
import numpy as np

# Başlık bastırmak için fonksiyon
def print_section_title(title):
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")

# Aktivasyon fonksiyonu ve türevi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Veriyi oku
try:
    data = pd.read_csv("thyroid_diff.csv")
except FileNotFoundError:
    print("CSV dosyası bulunamadı!")
    exit()

# Label encode (manuel olarak)
def label_encode(df, columns):
    encoders = {}
    for col in columns:
        unique_vals = df[col].unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df[col] = df[col].map(mapping)
        encoders[col] = mapping
    return df, encoders

label_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 
                 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 
                 'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred']

data, encoders = label_encode(data, label_columns)

# Giriş ve hedef verisi
X = data.drop(columns=['Recurred']).values
y = data['Recurred'].values.reshape(-1, 1)

# Veriyi normalize et
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Eğitimi ve testi elle ayıralım (80/20)
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Yapay sinir ağı parametreleri
input_size = X_train.shape[1]
hidden_size = 10
output_size = 1
learning_rate = 0.1
epochs = 500

# Ağırlıkları başlat
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Eğitim döngüsü
for epoch in range(epochs):
    # Forward
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Kayıp fonksiyonu (binary cross entropy değil ama basit MSE kullanıyoruz burada)
    loss = np.mean((y_train - A2) ** 2)

    # Backpropagation
    dA2 = -(y_train - A2)
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Ağırlık güncelle
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Tahmin fonksiyonu
def predict(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return (A2 > 0.5).astype(int)

# Başarı oranı
preds = predict(X_test)
accuracy = np.mean(preds == y_test)
print_section_title("Model Başarısı")
print(f"Test Verisi Doğruluğu: {accuracy * 100:.2f}%")
