import pandas as pd
import numpy as np

# Başlık yazdırma fonksiyonu
def print_section_title(title):
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")

# Kullanıcıdan veri alma ve doğrulama fonksiyonu
def get_user_input(prompt, valid_values=None, is_numeric=False):
    while True:
        value = input(prompt).strip()
        if is_numeric:
            try:
                value = int(value)
                if value < 0:
                    print("Lütfen pozitif bir sayı girin.")
                    continue
                return value
            except ValueError:
                print("Lütfen geçerli bir sayı girin.")
        else:
            if valid_values and value not in valid_values:
                print(f"Lütfen şu seçeneklerden birini girin: {', '.join(valid_values)}")
                continue
            return value

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

# Özellikler ve hedef
X = data.drop(columns=['Recurred']).values
y = data['Recurred'].values.reshape(-1, 1)

# Veriyi normalize et (0-1 aralığına getir)
X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

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
        return losses

# Modeli oluştur ve eğit
input_size = X_train.shape[1]  # Özellik sayısı
hidden_size = 10  # Gizli katmandaki nöron sayısı
output_size = 1  # Çıkış (ikili sınıflandırma: 0 veya 1)
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Modeli eğit
epochs = 1000
learning_rate}
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Modeli eğit
epochs = 1000
learning_rate = 0.01
losses = nn.train(X_train, y_train, epochs, learning_rate)

# Eğitim kaybı grafiğini çiz
import matplotlib.pyplot as plt
plt.plot(losses)
plt.title("Eğitim Kaybı (Loss)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Kullanıcıdan Yeni Kişi Verisi Al
print_section_title("Tiroid Kanser Tekrar Etme Tahmini")
print("Lütfen aşağıdaki bilgileri dikkatlice girin. Her sorunun yanında geçerli seçenekler belirtilecektir.")

# Veri setindeki kategorik sütunların geçerli değerlerini al
valid_values_dict = {col: list(le_dict[col].keys()) for col in label_columns if col in data.columns}

# Kullanıcıdan bilgileri al
new_person = {}
new_person['Age'] = get_user_input("Yaşınızı girin (örneğin: 45): ", is_numeric=True)
new_person['Gender'] = get_user_input(f"Cinsiyetiniz ({'/'.join(valid_values_dict['Gender'])}): ", valid_values=valid_values_dict['Gender'])
new_person['Smoking'] = get_user_input(f"Sigara kullanıyor musunuz? ({'/'.join(valid_values_dict['Smoking'])}): ", valid_values=valid_values_dict['Smoking'])
new_person['Hx Smoking'] = get_user_input(f"Geçmişte sigara kullandınız mı? ({'/'.join(valid_values_dict['Hx Smoking'])}): ", valid_values=valid_values_dict['Hx Smoking'])
new_person['Hx Radiothreapy'] = get_user_input(f"Geçmişte radyoterapi aldınız mı? ({'/'.join(valid_values_dict['Hx Radiothreapy'])}): ", valid_values=valid_values_dict['Hx Radiothreapy'])
new_person['Thyroid Function'] = get_user_input(f"Tiroid fonksiyonunuz ({'/'.join(valid_values_dict['Thyroid Function'])}): ", valid_values=valid_values_dict['Thyroid Function'])
new_person['Physical Examination'] = get_user_input(f"Fiziksel muayene sonucu ({'/'.join(valid_values_dict['Physical Examination'])}): ", valid_values=valid_values_dict['Physical Examination'])
new_person['Adenopathy'] = get_user_input(f"Adenopati var mı? ({'/'.join(valid_values_dict['Adenopathy'])}): ", valid_values=valid_values_dict['Adenopathy'])
new_person['Pathology'] = get_user_input(f"Patoloji sonucu ({'/'.join(valid_values_dict['Pathology'])}): ", valid_values=valid_values_dict['Pathology'])
new_person['Focality'] = get_user_input(f"Fokalite ({'/'.join(valid_values_dict['Focality'])}): ", valid_values=valid_values_dict['Focality'])
new_person['Risk'] = get_user_input(f"Risk seviyesi ({'/'.join(valid_values_dict['Risk'])}): ", valid_values=valid_values_dict['Risk'])
new_person['T'] = get_user_input(f"T evresi ({'/'.join(valid_values_dict['T'])}): ", valid_values=valid_values_dict['T'])
new_person['N'] = get_user_input(f"N evresi ({'/'.join(valid_values_dict['N'])}): ", valid_values=valid_values_dict['N'])
new_person['M'] = get_user_input(f"M evresi ({'/'.join(valid_values_dict['M'])}): ", valid_values=valid_values_dict['M'])
new_person['Stage'] = get_user_input(f"Kanser evresi ({'/'.join(valid_values_dict['Stage'])}): ", valid_values=valid_values_dict['Stage'])
new_person['Response'] = get_user_input(f"Tedavi yanıtı ({'/'.join(valid_values_dict['Response'])}): ", valid_values=valid_values_dict['Response'])

# Yeni kişinin verisini hazırla ve kodla
new_person_values = []
for col in data.drop(columns=['Recurred']).columns:
    new_person_values.append(new_person[col])
new_person_df = np.array(new_person_values, dtype=float).reshape(1, -1)

# Veriyi normalize et (eğitim verisiyle aynı şekilde)
new_person_df = (new_person_df - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)

# Tahmin yap
probability = nn.forward(new_person_df)
prediction = (probability > 0.5).astype(int)

# Sonuçları yazdır
print_section_title("Tahmin Sonuçları")
print(f"Girilen Bilgiler:\n{new_person}")
print(f"\nYineleme Yok Olasılığı: {(1 - probability[0][0]):.2%}")
print(f"Yineleme Var Olasılığı: {probability[0][0]:.2%}")
print(f"Sonuç: {'Yineleme Var' if prediction[0][0] == 1 else 'Yineleme Yok'}")