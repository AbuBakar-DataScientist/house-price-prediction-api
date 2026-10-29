# House Price Prediction API

A machine learning API built with **FastAPI** and **Random Forest Regressor** to predict house prices using California housing data.

## Features
- REST API for house price prediction
- Input validation using Pydantic
- Health check endpoint
- Pre-trained machine learning model

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- Pandas
- Joblib
- Uvicorn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR-USERNAME/house-price-prediction-api.git
   cd house-price-prediction-api
   ```

2. Install dependencies:

   ```bash
   pip install fastapi uvicorn pandas scikit-learn joblib
   ```

3. Start the API:

   ```bash
   uvicorn main:app --reload
   ```

4. Open the interactive API documentation:

   http://127.0.0.1:8000/docs

## API Endpoints

- `GET /` — Welcome message
- `GET /health` — API health status
- `POST /predict` — Predict house price

## Model Files

Place `house_price_model.joblib` and `house_price_columns.joblib` in the project root before starting the API.

## Author

Abu Bakar Siddique

GitHub: https://github.com/YOUR-USERNAME
