from fastapi import FastAPI
from predict import router as predict_router

app = FastAPI(
    title="Meli Fraud API",
    description="API developved for Meli test",
    version="1.0.0",
)

app.include_router(predict_router, prefix="/predict", tags=["predict"])


@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API"}
