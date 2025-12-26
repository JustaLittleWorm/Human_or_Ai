import hashlib
from app.services.predictor_base import PredictorBase

class MockPredictor(PredictorBase):
    def predict(self, text: str):
        # Deterministic “fake” score: sunum/test için stabil
        h = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        ai1 = (h % 10000) / 100.0
        ai2 = ((h // 7) % 10000) / 100.0
        ai3 = ((h // 13) % 10000) / 100.0

        models = [
            {"name": "model1", "ai": round(ai1, 2), "human": round(100 - ai1, 2)},
            {"name": "model2", "ai": round(ai2, 2), "human": round(100 - ai2, 2)},
            {"name": "model3", "ai": round(ai3, 2), "human": round(100 - ai3, 2)},
        ]
        avg_ai = round((ai1 + ai2 + ai3) / 3.0, 2)
        return {
            "input_chars": len(text),
            "models": models,
            "avg": {"ai": avg_ai, "human": round(100 - avg_ai, 2)}
        }
