import streamlit as st
import joblib
import numpy as np
import pandas as pd

if "history" not in st.session_state:
    st.session_state.history = []

# Sayfa ayarları
st.set_page_config(
    page_title="Human vs AI Text Classification",
    layout="wide"
)

st.title("Human or AI ")
st.write("Metnin insan mı yoksa yapay zekâ tarafından mı yazıldığını tahmin eder.")

# Modelleri yükle
vectorizer = joblib.load(r"D:\Masaüstü\human_ai\models\vectorizer.pkl")

models = {
    
    "Linear SVM(Önerilir)": joblib.load(r"D:\Masaüstü\human_ai\models\svm.pkl"),
    "Logistic Regression": joblib.load(r"D:\Masaüstü\human_ai\models\logreg.pkl"),
    "Random Forest": joblib.load(r"D:\Masaüstü\human_ai\models\rf.pkl")
}

label_map = {0: "Human", 1: "AI"}


tab1, tab2,tab3 = st.tabs(["Human or AI ", "Geçmiş","Geliştiriciler"])


with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Makale giriniz")
        user_text = st.text_area(
            "Metin",
            height=300,
            placeholder="Makaleyi buraya giriniz...",
            label_visibility="collapsed"
        )

        model_name = st.selectbox(
            "Model Seciniz",
            list(models.keys())
        )

        submit = st.button("Tahmin")
        clear = st.button("Temizle")

with col2:
        st.subheader("Tahmin Sonuçları")

        if submit and user_text.strip():
            model = models[model_name]

            X_vec = vectorizer.transform([user_text])
            proba = model.predict_proba(X_vec)[0]
            pred = np.argmax(proba)

            st.success(f"Tip: {label_map[pred]}")
            st.session_state.history.append({
                "Time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Model": model_name,
                "Prediction": label_map[pred],
                "AI %": round(proba[1]*100, 2),
                "Human %": round(proba[0]*100, 2),
                "Text (Preview)": user_text[:150] + "..."
            })
            
            st.metric("AI Olasılığı", f"%{proba[1]*100:.2f}")
            st.metric("Human Olasılığı", f"%{proba[0]*100:.2f}")


            prob_df = pd.DataFrame({
                "Class": ["Human", "AI"],
                "Probability": [proba[0], proba[1]]
            })

            st.bar_chart(prob_df.set_index("Class"))

        if clear:
            st.rerun()
with tab2:
    st.subheader("Geçmiş Tahmin Sonuçları")
    if st.button("Geçmişi Temizle"):
        st.session_state.history = []
        st.rerun()
    if len(st.session_state.history) == 0:
            st.info("Henüz tahmin yapılmadı.")
    else:
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df, use_container_width=True)
with tab3:
    st.subheader("İsim Soyisim")
    _1="Burak Ali Akbulut"
    _2="Mert Enes Tomak"
    _3="Tuanna Ertuğ"
    _4="İrem Koyuncu"
    st.markdown(f"""
    {_1}  
    {_2}  
    {_3}  
    {_4}  
    """)
