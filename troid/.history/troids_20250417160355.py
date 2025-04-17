import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from tabulate import tabulate

# Başlık yazdırma fonksiyonu
def print_section_title(title):
    print("\n" + "=" * 50)
    print(f" {title} ")
    print("=" * 50 + "\n")

# 1. Veri Setini Yükleyin
data = pd.read_csv('thyroid_diff.csv')

# 2. Kategorik Verileri Kodlayın
label_columns = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 'Thyroid Function', 
                 'Physical Examination', 'Adenopathy', 'Pathology', 'Focality', 
                 'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred']
le = LabelEncoder()
for col in label_columns:
    if col in data.columns:
        data[col] = le.fit_transform(data[col])

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

# 7. Tahmin Yapın
y_pred = model.predict(X_test)

# 8. Sonuçları Değerlendirin
print_section_title("Model Performans Sonuçları")

# Karmaşıklık Matrisi
print("Karmaşıklık Matrisi:")
cm = confusion_matrix(y_test, y_pred)
cm_table = [["", "Tahmin: Yineleme Yok", "Tahmin: Yineleme Var"], 
            ["Gerçek: Yineleme Yok", cm[0][0], cm[0][1]], 
            ["Gerçek: Yineleme Var", cm[1][0], cm[1][1]]]
print(tabulate(cm_table, headers='firstrow', tablefmt='psql'))

# Sınıflandırma Raporu
print("\nSınıflandırma Raporu:")
report = classification_report(y_test, y_pred, target_names=["Yineleme Yok (0)", "Yineleme Var (1)"])
report_lines = report.split('\n')
formatted_report = []
for line in report_lines:
    if line.strip():
        parts = line.split()
        if len(parts) >= 5:
            formatted_report.append(parts)
        elif "accuracy" in line:
            formatted_report.append(["Doğruluk", "", "", parts[-2], parts[-1]])
        elif "avg" in line:
            formatted_report.append(parts)

print(tabulate(formatted_report, headers=["Sınıf", "Kesinlik", "Duyarlılık", "F1-Skoru", "Destek"], tablefmt='psql'))