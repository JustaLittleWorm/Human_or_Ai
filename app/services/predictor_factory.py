# app/services/predictor_factory.py
from app.settings import settings
from app.services.mock_predictor import MockPredictor
from app.services.real_predictor import RealPredictor


def get_predictor():
    if settings.use_mock:
        return MockPredictor()
    return RealPredictor()
