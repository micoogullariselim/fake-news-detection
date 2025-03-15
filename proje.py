import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib


# Sahte ve gerçek haberleri yükleme
data_sahte = pd.read_csv('Fake.csv')
data_doğru = pd.read_csv('True.csv')
data_sahte["class"] = 0  
data_doğru["class"] = 1
data_birlesik = pd.concat([data_sahte, data_doğru], axis=0)




# Veri işleme fonksiyonu
def metin_düzenleme(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

# Metin verisini işleme
data_birlesik['text'] = data_birlesik['text'].apply(metin_düzenleme)
x = data_birlesik['text']
y = data_birlesik['class']

# Eğitim ve test verisine ayırma
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# TF-IDF ile vektörleştirme
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Modelleri eğitme
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)

NB = MultinomialNB()
NB.fit(xv_train, y_train)

SVM = SVC(kernel='linear')
SVM.fit(xv_train, y_train)


# Tahmin yapma ve doğruluk hesaplama fonksiyonu
def calculate_accuracy(predictions, true_labels):
    correct = sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = (correct / total) * 100
    return accuracy, total - correct  # (Doğru Yüzde, Yanlış Sayısı)

# Tahminler ve doğruluk değerlerini hesaplama
pred_lr = LR.predict(xv_test)
pred_dt = DT.predict(xv_test)
pred_rf = RF.predict(xv_test)
pred_nb = NB.predict(xv_test)
pred_svm = SVM.predict(xv_test)

print(pred_lr)
print(pred_dt)
print(pred_rf)
print(pred_nb)
print(pred_svm)

accuracy_lr, wrong_lr = calculate_accuracy(pred_lr, y_test)
accuracy_dt, wrong_dt = calculate_accuracy(pred_dt, y_test)
accuracy_rf, wrong_rf = calculate_accuracy(pred_rf, y_test)
accuracy_nb, wrong_nb = calculate_accuracy(pred_nb, y_test)
accuracy_svm, wrong_svm = calculate_accuracy(pred_svm, y_test)

# Sonuçları yazdırma
print("Doğru Yüzde ve Yanlış Sayıları:")
print(f"Logistic Regression: Doğru: {accuracy_lr:.2f}%, Yanlış: {wrong_lr}")
print(f"Decision Tree: Doğru: {accuracy_dt:.2f}%, Yanlış: {wrong_dt}")
print(f"Random Forest: Doğru: {accuracy_rf:.2f}%, Yanlış: {wrong_rf}")
print(f"Naive Bayes: Doğru: {accuracy_nb:.2f}%, Yanlış: {wrong_nb}")
print(f"SVM: Doğru: {accuracy_svm:.2f}%, Yanlış: {wrong_svm}")

# Modelleri kaydetme
joblib.dump(vectorization, 'vectorizer.pkl')
joblib.dump(LR, 'LogisticRegression_model.pkl')
joblib.dump(DT, 'DecisionTree_model.pkl')
joblib.dump(RF, 'RandomForest_model.pkl')
joblib.dump(NB, 'NaiveBayes_model.pkl')
joblib.dump(SVM, 'SVM_model.pkl') 

print("Modeller ve vektörleştirici başarıyla kaydedildi.")
