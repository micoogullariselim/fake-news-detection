from flask import Flask, request, jsonify, render_template
from googletrans import Translator
from langdetect import detect
from collections import Counter
import re
import joblib
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from langcodes import Language
import langcodes
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd


#NLTK Stop Words
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

newsData = []
app = Flask(__name__)


# Model ve vektörleştiriciyi yükleme
vectorization = joblib.load('vectorizer.pkl')
LR = joblib.load('LogisticRegression_model.pkl')
DT = joblib. load( 'DecisionTree_model.pkl')
RF = joblib.load('RandomForest_model.pkl')
NB = joblib.load( 'NaiveBayes_model.pkl')
SVM = joblib.load('SVM_model.pkl')


translator = Translator()


# Anahtar kelime çikarzmı fonksiyonu
def anahtar_kelime(text, num_keywords=5):
   words = re.findall(r'\b\w+\b', text.lower())
   filtered_words = [word for word in words if word not in stop_words]
   tfidf_vectorizer = TfidfVectorizer(max_features=1000)
   tfidf_matrix = tfidf_vectorizer.fit_transform([' '.join(filtered_words)])
   feature_names = tfidf_vectorizer.get_feature_names_out()
   tfidf_scores = zip(feature_names, tfidf_matrix.toarray().flatten())
   sorted_keywords = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
   keywords = [word for word, score in sorted_keywords[:num_keywords]]
   return keywords



# Haber kelime frekansı grafiği oluşturma
def plot_word_frequency(common_words):
      words, counts = zip(*common_words)
      plt.figure(figsize=(8, 5))
      plt.bar(words, counts, color='skyblue')
      plt.title('Haberde En Çok Kullanblan Kelimeler')
      plt.xlabel('Kelimeler')
      plt.ylabel('Frekans')
      plt.tight_layout()


      # Grafik dosyasını static dizinine kaydet
      plt.savefig('static/common_words.png')
      plt.close() # Görsellestirme kapanışını yap


data_sahte = pd.read_csv('Fake.csv')
data_doğru = pd.read_csv('True.csv')


@app. route('/')
def home():

    return render_template('index.html')


@app.route('/analyze', methods=['GET'])
def analyze_get():
     return render_template('analiz.html')


@app.route('/analyze', methods=['POST'])
def analyze_news():
       data = request.get_json()
       news = data.get('news')


       # Haber dilini tespit et
       detected_lang = detect(news)


       if langcodes.Language(detected_lang).is_valid():
          dil = langcodes.Language(detected_lang).display_name()
          dil_turkce = translator.translate(dil, src=detected_lang, dest='tr').text
          if detected_lang == "tr":
             dil_turkce = "Türkçe"
       else:
          dil = detected_lang
          dil_turkce = "Bilinmeyen Dil"

        
        # Haberi İngilizceye çevir
       if detected_lang != 'en':
           news = translator.translate(news, src=detected_lang, dest='en').text


       if detected_lang != 'tr':
          translated_to_tr = translator.translate(news, src=detected_lang, dest='tr').text
       else:
          translated_to_tr = news


        # Metni vektörleştir ve modelle tahmin yap
       new_xv_test = vectorization.transform([news])
       lr_prediction = LR.predict(new_xv_test)[0]
       dt_prediction = DT.predict(new_xv_test)[0]
       rf_prediction = RF.predict(new_xv_test)[0]
       nb_prediction = NB.predict(new_xv_test)[0]
       svm_prediction = SVM.predict(new_xv_test)[0]


       #Sonuçları değerlendirme
       lr_label = "Sahte Haber" if lr_prediction == 0 else "Sahte Değil"
       dt_label = "Sahte Haber" if dt_prediction == 0 else "Sahte Değil"
       rf_label = "Sahte Haber" if rf_prediction == 0 else "Sahte Değil"
       nb_label = "Sahte Haber" if nb_prediction == 0 else "Sahte Değil"
       svm_label = "Sahte Haber" if svm_prediction == 0 else "Sahte Değil"


       # Doğru/yanlıs yüzdelerini hesapla
       predictions = [lr_prediction, dt_prediction, rf_prediction, nb_prediction, svm_prediction]
       correct_predictions = sum(predictions) # Sahte degil = 1, Sahte = 0
       total_predictions = len(predictions)


       accuracy = (correct_predictions / total_predictions) * 100
       error_rate = 100 - accuracy


       # Haber analizi
       news_words = re.findall(r'\b\w+\b', news.lower())
       filtered_words = [word for word in news_words if word not in stop_words]
       word_freq = Counter(filtered_words)
       common_words = word_freq.most_common(5)


       #Grafik olustur
       plot_word_frequency(common_words)


       # İçerik analizi (duygu analizi)
       analiz = TextBlob(news)
       sentiment = analiz.sentiment


       # Anahtar kelimeler
       keywords = anahtar_kelime(news)



       # Sonucu JSON olarak döndür
       result = {
       "lr_label": lr_label,
       "dt_label": dt_label,
       "rf_label": rf_label,
       "nb_label": nb_label,
       "svm_label": svm_label,
       "haber_dili": dil,
       "haber_dili_tr": dil_turkce,
       "accuracy": accuracy,
       "error_rate": error_rate,
       "common_words": common_words,
       "keywords": keywords,
       "sentiment":{
       "polarity": sentiment.polarity,
       "subjectivity": sentiment.subjectivity
       },
       "news": news,
       "translated_to_tr": translated_to_tr,
       "word_ frequency _chart":common_words,

       }
       
       newsData.append(result)
       return jsonify(result)




if __name__ == '__main__':
    app.run(debug=True)
