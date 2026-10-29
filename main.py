import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI()

model = joblib.load("house_price_model.joblib")
features = joblib.load("house_price_columns.joblib")

class HouseFeatures(BaseModel):
    MedInc: float = Field(..., description="Median income in block group") # (...) Ellipsis kehte hain.
    # Pydantic mein iska matlab hai ke ye field required hai.
    # user ko value deni hi hogi. Agar value missing ho, to validation error aayega.
    HouseAge: float = Field(..., description="Median house age in block group")
    AveRooms: float = Field(..., description="Average number of rooms per household")
    AveBedrms: float = Field(..., description="Average number of bedrooms per household")
    Population: float = Field(..., description="Block group population")
    AveOccup: float = Field(..., description="Average number of household members")
    Latitude: float = Field(..., description="Block group latitude")
    Longitude: float = Field(..., description="Block group longitude")

# home
@app.get("/")
def home():
    return {
        "message": "Welcome to the House Price Prediction API",
        "Status": "Running",
        "endpoint": "send a POST request to /predict."
        }
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "Random Forest Regressor",
        "features": features,
        "avg_error": "$32,754"
        }

@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms,
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])
        prediction = model.predict(input_data)
        return {"predicted_price": prediction[0] * 100000}  # Convert to actual price
    except Exception as e:
        raise HTTPException(
            status_code=400, 
            detail=str(e)
            )