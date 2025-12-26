# app/api.py
from fastapi import APIRouter, HTTPException
from app.schemas import PredictRequest, PredictResponse
from app.services.predictor_factory import get_predictor

router = APIRouter()
predictor = get_predictor()


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Empty text")

    result = predictor.predict(text)
    return result


@router.get("/health")
def health():
    return {"status": "ok"}
