from fastapi import FastAPI
from app.api import router
from app.middleware import TimingMiddleware

app = FastAPI(title="Human or AI - Abstract Detector")
app.add_middleware(TimingMiddleware)
app.include_router(router)
