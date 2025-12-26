def predict_text(text: str, vectorizer, model):
    """
    Model tahmini yapan çekirdek fonksiyon.
    White-box testler bu fonksiyonu hedef alır.
    """
    X = vectorizer.transform([text])
    proba = model.predict_proba(X)[0]
    return proba
