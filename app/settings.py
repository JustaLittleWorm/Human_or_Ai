# app/settings.py
from dataclasses import dataclass

@dataclass
class Settings:
    use_mock: bool = False   # True yaparsan MockPredictor çalışır
    max_chars: int = 20000

settings = Settings()
