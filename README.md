Human or AI Makale Sınıfladırma
Proje birden fazla model kullanarak akademik metinler için "insan mı?" "yapay zeka mı?" yazdı sorusuna cevap aramakta.

Proje Yapısı
app/: Streamlit web application
models/: Trained ML models and vectorizer
data/: Raw and processed datasets
notebooks/: EDA, preprocessing, and training notebooks
docs/: Project documentation
Gereksinimler
Terminal'den; pip install -r requirements.txt Komutu ile gerekli gereksinimleri yükleyebilir,

python -m streamlit run app/app.py komutu ile çalıştırabilirsiniz.

cd ...human_ai\app dizinine girmeyi unutmayın ;D

Test
Statik kod analizi SonarQube Cloud kullanılarak gerçekleştirildi. Proje, sıfır kritik, büyük veya küçük sorunla Kalite Kontrol Aşamasını başarıyla geçti.

##👤 Geliştiriciler: Burak Ali Akbulut,Mert Enes Tomak,Tuanna Ertuğ,İrem Koyuncu
