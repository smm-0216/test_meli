import joblib

import pandas as pd
from fastapi import APIRouter
from sqlalchemy import create_engine

from feature_engineering import FeatureEngineering
from schemas import PredictRequest, PredictResponse

router = APIRouter()

kept_categories = {"A", "B", "C"}
final_columns = ["A", "B", "F", "G", "Q", "R", "Monto", "J_B", "J_C"]

fe = FeatureEngineering(
    encoder_path="artifacts/onehot_encoder.joblib",
    kept_categories=kept_categories,
    final_columns=final_columns,
)
model = joblib.load("artifacts/model.joblib")


def save_data(data_fe, prediction):
    data_to_db = pd.DataFrame(data_fe)
    data_to_db['prediction'] = prediction
    data_to_db['created_at'] = pd.Timestamp.now()
    engine = create_engine("sqlite:///meli.db", echo=False)
    data_to_db.to_sql('deployment', con=engine, if_exists='append', index=False)


@router.post("/", response_model=PredictResponse)
def make_prediction(request: PredictRequest):
    data = request.model_dump()
    data_fe = fe.transform(data)
    prediction = model.predict(pd.DataFrame(data_fe))
    save_data(data_fe, prediction)
    return PredictResponse(prediction=prediction)
