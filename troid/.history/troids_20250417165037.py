# ... veri yükleme ve label encoding kısmı aynı kalıyor ...

# ---------- Normalize sadece eğitim seti üzerinden ----------
X_full = data.drop(columns=['Recurred']).values
y = data['Recurred'].values.reshape(-1, 1)

split = int(0.8 * len(X_full))
X_train_raw, X_test_raw = X_full[:split], X_full[split:]
y_train, y_test = y[:split], y[split:]

X_min = X_train_raw.min(axis=0)
X_max = X_train_raw.max(axis=0)

X_train = (X_train_raw - X_min) / (X_max - X_min + 1e-8)
X_test = (X_test_raw - X_min) / (X_max - X_min + 1e-8)

# ---------- Ağ Parametreleri ----------
input_size = X_train.shape[1]
hidden1 = 16
hidden2 = 12
output_size = 1
epochs = 1000
lr = 0.05

# ---------- Ağırlıklar ----------
np.random.seed(42)
W1 = np.random.randn(input_size, hidden1) * np.sqrt(2./input_size)
b1 = np.zeros((1, hidden1))

W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2./hidden1)
b2 = np.zeros((1, hidden2))

W3 = np.random.randn(hidden2, output_size) * np.sqrt(2./hidden2)
b3 = np.zeros((1, output_size))

losses = []

# ---------- Aktivasyon Fonksiyonları ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return x * (1 - x)

# ---------- Eğitim Döngüsü ----------
for epoch in range(epochs):
    # FORWARD
    Z1 = np.dot(X_train, W1) + b1
    A1 = sigmoid(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    # KAYIP
    loss = np.mean((y_train - A3)**2)
    losses.append(loss)

    # BACKPROP
    dA3 = -(y_train - A3)
    dZ3 = dA3 * sigmoid_deriv(A3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = np.sum(dZ3, axis=0, keepdims=True)

    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_deriv(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_deriv(A1)
    dW1 = np.dot(X_train.T, dZ1)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Ağırlık güncelle
    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# ---------- Test ----------
Z1_test = np.dot(X_test, W1) + b1
A1_test = sigmoid(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = sigmoid(Z2_test)

Z3_test = np.dot(A2_test, W3) + b3
A3_test = sigmoid(Z3_test)

preds = (A3_test > 0.5).astype(int)
acc = np.mean(preds == y_test)
print(f"\n✅ Test Doğruluğu: {acc * 100:.2f}%")

# ---------- Loss Grafiği ----------
plt.plot(losses)
plt.title("Loss Grafiği")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# ---------- Kullanıcıdan Veri Al ve Tahmin Et ----------
def get_user_input(label_encoders):
    print("\n🧪 Yeni hasta bilgisi girin:")
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
    person = (person - X_min) / (X_max - X_min + 1e-8)

    # Tahmin
    z1 = np.dot(person, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    a3 = sigmoid(z3)

    print(f"\n📈 Yineleme Olasılığı: %{a3[0][0]*100:.2f}")
    print("📌 Tahmin: ", "Yineleme Var" if a3[0][0] > 0.5 else "Yineleme Yok")

get_user_input(label_encoders)
