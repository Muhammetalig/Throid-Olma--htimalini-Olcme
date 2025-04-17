import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
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

# 1. Veri Setini Yükleyin
try:
    data = pd.read_csv('thyroid_diff.csv')
except FileNotFoundError:
    print("Hata: 'thyroid_diff.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# 2. Kategorik Verileri Kodlayın ve LabelEncoder nesnelerini saklayın
label_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 
                 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 
                 'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred']
le_dict = {}  # Her sütun için LabelEncoder nesnesini saklamak
for col in label_columns:
    if col in data.columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        le_dict[col] = le  # LabelEncoder nesnesini sakla

# 3. Özellikler ve Hedef
X = data.drop(columns=['Recurred'])
y = data['Recurred']

# 4. Veriyi Bölün
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. SMOTE ile Dengesizliği Giderin
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 6. Modeli Eğitin
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight={0: 1, 1: 3})
model.fit(X_train, y_train)

# 7. Kullanıcıdan Yeni Kişi Verisi Al
print_section_title("Tiroid Kanser Tekrar Etme Tahmini")
print("Lütfen aşağıdaki bilgileri dikkatlice girin. Her sorunun yanında geçerli seçenekler belirtilecektir.")

# Veri setindeki kategorik sütunların geçerli değerlerini al
valid_values_dict = {col: list(le_dict[col].classes_) for col in label_columns if col in data.columns}

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

# 8. Yeni kişinin verisini DataFrame'e çevir ve kodla
new_person_df = pd.DataFrame([new_person])
for col in label_columns[:-1]:  # Recurred hariç
    if col in new_person_df.columns:
        new_person_df[col] = le_dict[col].transform(new_person_df[col])

# 9. Tahmin yap
probabilities = model.predict_proba(new_person_df)
prediction = model.predict(new_person_df)

# 10. Sonuçları yazdır
print_section_title("Tahmin Sonuçları")
print(f"Girilen Bilgiler:\n{new_person}")
print(f"\nYineleme Yok Olasılığı: {probabilities[0][0]:.2%}")
print(f"Yineleme Var Olasılığı: {probabilities[0][1]:.2%}")
print(f"Sonuç: {'Yineleme Var' if prediction[0] == 1 else 'Yineleme Yok'}")