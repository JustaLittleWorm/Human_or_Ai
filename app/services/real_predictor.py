# app/services/real_predictor.py
from dataclasses import dataclass
from typing import Dict, Any, List

import joblib
import numpy as np
from pathlib import Path

from app.services.predictor_base import PredictorBase


@dataclass
class ModelOut:
    name: str
    ai: float
    human: float


class RealPredictor(PredictorBase):
    def __init__(self) -> None:
        """
        logreg.pkl, rf.pkl, svm.pkl, vectorizer.pkl
        human_ai/human_ai/models klasöründe.
        """
        # .../app/services/real_predictor.py -> .../human_ai/human_ai
        project_root = Path(__file__).resolve().parents[2]
        models_dir = project_root / "models"

        self.vectorizer = joblib.load(models_dir / "vectorizer.pkl")
        self.model1 = joblib.load(models_dir / "logreg.pkl")
        self.model2 = joblib.load(models_dir / "rf.pkl")
        self.model3 = joblib.load(models_dir / "svm.pkl")

        # AI sınıfının index’ini model.classes_ içinden otomatik bulmaya çalış
        # Eğitimde label'lar 0/1 veya "ai"/"human" ise çoğu durumda otomatik çalışır.
        self.ai_index_m1 = self._detect_ai_index(self.model1)
        self.ai_index_m2 = self._detect_ai_index(self.model2)
        self.ai_index_m3 = self._detect_ai_index(self.model3)

    def _detect_ai_index(self, model) -> int:
        """
        model.classes_ içinden AI sınıfının index'ini bulmaya çalışır.
        Gerekirse burada elle sabitleyebilirsin.
        """
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            # Ör: [0, 1] veya ['human', 'ai']
            for i, c in enumerate(classes):
                if c in (1, "ai", "AI", "Ai", "AI_TEXT"):
                    return i
            # Hiçbiri eşleşmediyse son index'i AI kabul et (fallback)
            return len(classes) - 1
        # classes_ yoksa binary varsayalım, AI = 1
        return 1

    def _score_ai(self, model, X, ai_index: int) -> float:
        """
        0.0–1.0 arası AI olasılığı döner.
        """
        # 1) predict_proba varsa en net
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            return float(proba[ai_index])

        # 2) decision_function varsa sigmoid ile olasılığa çevir
        if hasattr(model, "decision_function"):
            d = float(model.decision_function(X)[0])
            return float(1.0 / (1.0 + np.exp(-d)))

        # 3) En kötü ihtimal: predict sınıfını kullan
        pred = model.predict(X)[0]
        if pred == ai_index or pred in ("ai", "AI", 1):
            return 1.0
        return 0.0

    def predict(self, text: str) -> Dict[str, Any]:
        # Eğitimde nasıl yaptıysanız aynı şekilde: burada ben direkt ham metni veriyorum.
        X = self.vectorizer.transform([text])

        models: List[ModelOut] = []

        for name, model, idx in [
            ("logreg", self.model1, self.ai_index_m1),
            ("rf",     self.model2, self.ai_index_m2),
            ("svm",    self.model3, self.ai_index_m3),
        ]:
            ai_prob = self._score_ai(model, X, idx) * 100.0
            human_prob = 100.0 - ai_prob
            models.append(
                ModelOut(
                    name=name,
                    ai=round(ai_prob, 2),
                    human=round(human_prob, 2),
                )
            )

        avg_ai = round(sum(m.ai for m in models) / len(models), 2)
        avg_human = round(100.0 - avg_ai, 2)

        return {
            "input_chars": len(text),
            "models": [m.__dict__ for m in models],
            "avg": {"ai": avg_ai, "human": avg_human},
        }
