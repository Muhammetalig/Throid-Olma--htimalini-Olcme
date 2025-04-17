import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Başlık yazdırma fonksiyonu
def print_section_title(title):
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")

# Aktivasyon fonksiyonları
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def sigmoid_derivative(x):
    return x * (1 - x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Veri setini yükle ve hazırla
try:
    data = pd.read_csv('thyroid_diff.csv')
except FileNotFoundError:
    print("Hata: 'thyroid_diff.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# Kategorik verileri kodla (Label Encoding)
label_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 
                 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 
                 'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred']
le_dict = {}  # Her sütun için LabelEncoder nesnesini saklamak
for col in label_columns:
    if col in data.columns:
        unique_values = data[col].unique()
        le_dict[col] = {val: idx for idx, val in enumerate(unique_values)}
        data[col] = data[col].map(le_dict[col])

# Hata ayıklama: le_dict içeriğini kontrol et
print("le_dict içeriği:")
for col in le_dict:
    print(f"{col}: {le_dict[col]}")

# Özellikler ve hedef
X = data.drop(columns=['Recurred']).values
y = data['Recurred'].values.reshape(-1, 1)

# Sınıf dengesizliğini kontrol et
print("Sınıf dağılımı (Yineleme Yok/Yineleme Var):", np.bincount(y.flatten().astype(int)))

# Manuel olarak sınıf dengesizliğini ele al (oversampling)
# Yineleme Var (1) sınıfını çoğalt
X_0 = X[y.flatten() == 0]  # Yineleme Yok
y_0 = y[y.flatten() == 0]
X_1 = X[y.flatten() == 1]  # Yineleme Var
y_1 = y[y.flatten() == 1]

# Yineleme Var sınıfını 3 katına çıkar
X_1_oversampled = np.repeat(X_1, 3, axis=0)
y_1_oversampled = np.repeat(y_1, 3, axis=0)

# Yeni dengelenmiş veri setini oluştur
X = np.vstack((X_0, X_1_oversampled))
y = np.vstack((y_0, y_1_oversampled))

print("Dengelenmiş sınıf dağılımı (Yineleme Yok/Yineleme Var):", np.bincount(y.flatten().astype(int)))

# Veriyi normalize et (0-1 aralığına getir)
X_min = X.min(axis=0)
X_max = X.max(axis=0)
X = (X - X_min) / (X_max - X_min + 1e-8)

# Eğitim ve test setine böl
np.random.seed(42)
indices = np.random.permutation(X.shape[0])
train_size = int(0.8 * X.shape[0])
train_indices = indices[:train_size]
test_indices = indices[train_size:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Yapay Sinir Ağı Sınıfı
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Ağırlıkları ve bias'ları başlat
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Girişten gizli katmana ağırlıklar
        self.b1 = np.zeros((1, hidden_size))  # Gizli katman bias
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Gizli katmandan çıkışa ağırlıklar
        self.b2 = np.zeros((1, output_size))  # Çıkış katmanı bias

    def forward(self, X):
        # İleri yayılım
        self.Z1 = np.dot(X, self.W1) + self.b1  # Girişten gizli katmana
        self.A1 = relu(self.Z1)  # Gizli katman aktivasyonu
        self.Z2 = np.dot(self.A1, self.W2) + self.b2  # Gizli katmandan çıkışa
        self.A2 = sigmoid(self.Z2)  # Çıkış katmanı aktivasyonu
        return self.A2

    def backward(self, X, y, output, learning_rate):
        # Geri yayılım
        self.error = y - output
        self.delta2 = self.error * sigmoid_derivative(output)

        self.error_hidden = np.dot(self.delta2, self.W2.T)
        self.delta1 = self.error_hidden * relu_derivative(self.A1)

        # Ağırlıkları ve bias'ları güncelle
        self.W2 += learning_rate * np.dot(self.A1.T, self.delta2)
        self.b2 += learning_rate * np.sum(self.delta2, axis=0, keepdims=True)
        self.W1 += learning_rate * np.dot(X.T, self.delta1)
        self.b1 += learning_rate * np.sum(self.delta1, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        losses = []
        for epoch in range(epochs):
            # İleri yayılım
            output = self.forward(X)
            # Kayıp hesapla (binary cross-entropy)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            losses.append(loss)
            # Geri yayılım
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Kayıp: {loss:.4f}")
                # Test seti üzerinde doğruluk hesapla
                test_output = self.forward(X_test)
                test_predictions = (test_output > 0.5).astype(int)
                accuracy = np.mean(test_predictions == y_test)
                print(f"Test Doğruluğu: {accuracy:.4f}")
        return losses

# Modeli oluştur ve eğit
input_size = X_train.shape[1]  # Özellik sayısı
hidden_size = 16  # Gizli katmandaki nöron sayısı
output_size = 1  # Çıkış (ikili sınıflandırma: 0 veya 1)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Modeli eğit
epochs = 2000
learning_rate = 0.005
losses = nn.train(X_train, y_train, epochs, learning_rate)

# Eğitim kaybı grafiğini çiz
plt.plot(losses)
plt.title("Eğitim Kaybı (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Örnek Hasta Oluştur
print_section_title("Örnek Hasta için Tiroid Kanser Tekrar Etme Tahmini")

# Örnek hasta verileri (le_dict'e uygun olarak seçildi)
new_person = {
    'Age': 35,
    'Gender': 'Female',
    'Smoking': 'No',
    'Hx Smoking': 'No',
    'Hx Radiothreapy': 'No',
    'Thyroid Function': 'Euthyroid',
    'Physical Examination': 'Single nodular goiter-left',
    'Adenopathy': 'No',
    'Pathology': 'Papillary',
    'Focality': 'Uni-Focal',
    'Risk': 'Low',
    'T': 'T2',
    'N': 'N0',
    'M': 'M0',
    'Stage': 'I',
    'Response': 'Excellent'
}

# Yeni kişinin verisini hazırla ve kodla
new_person_values = []
for col in data.drop(columns=['Recurred']).columns:
    if col == 'Age':
        new_person_values.append(new_person[col])  # Yaş zaten sayısal
    else:
        # Kategorik veriyi sayısal değere çevir
        if new_person[col] in le_dict[col]:
            encoded_value = le_dict[col][new_person[col]]
            new_person_values.append(encoded_value)
        else:
            print(f"Hata: '{col}' için '{new_person[col]}' değeri le_dict'te bulunamadı. Geçerli değerler: {le_dict[col]}")
            exit()

# Hata ayıklama: Kodlanmış değerleri kontrol et
print("Kodlanmış new_person_values:", new_person_values)

# Veriyi numpy array'e çevir ve normalize et
new_person_df = np.array(new_person_values, dtype=float).reshape(1, -1)
new_person_df = (new_person_df - X_min) / (X_max - X_min + 1e-8)

# Tahmin yap
probability = nn.forward(new_person_df)
prediction = (probability > 0.5).astype(int)

# Hata ayıklama: Çıkış değerlerini kontrol et
print("Çıkış olasılığı (sigmoid sonrası):", probability)

# Sonuçları yazdır
print_section_title("Tahmin Sonuçları")
print(f"Örnek Hasta Bilgileri:\n{new_person}")
print(f"\nYineleme Yok Olasılığı: {(1 - probability[0][0]):.2%}")
print(f"Yineleme Var Olasılığı: {probability[0][0]:.2%}")
print(f"Sonuç: {'Yineleme Var' if prediction[0][0] == 1 else 'Yineleme Yok'}")