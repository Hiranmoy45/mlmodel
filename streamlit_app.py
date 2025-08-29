from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
import uvicorn
import os
import gdown
import joblib
from train_recommendation import get_recommendations
import json
from utils import enrich_properties
# FastAPI app
app = FastAPI(title="Real Estate Recommendation API")

# Enable CORS for frontend
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Google Drive file IDs
files = {
    "price-model.joblib": "1pSLEqZ2aqnPCVGgEwf8It85WxJzUAzM4",
    "preprocessor.pkl": "1zFpvtuSJEvqoj-PSQB9Y9Xq4xEeS7cRV",
    "processed_df.pkl": "17N4gpO7eoFU0f1TyAIPbN9OJfnoobgI2",
    "feature_matrix.pkl": "14qvItow972i41a9QhRiqXAc0ObwtDRQ-"
}

# Download if not exists
for filename, file_id in files.items():
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {filename}...")
        gdown.download(url, filename, quiet=False)


# Load model artifacts
try:
    price_pipe = joblib.load("price-model.joblib")
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    with open("processed_df.pkl", "rb") as f:
        df = pickle.load(f)
    with open("feature_matrix.pkl", "rb") as f:
        feature_matrix = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    print("Please run `python train_and_predict.py` first.")
    preprocessor, df, feature_matrix = None, None, None


class PredictRequest(BaseModel):
    total_area: float
    bhk: int
    baths: int
    balcony: bool
    location: str

# Pydantic model for wishlist item
class WishlistItem(BaseModel):
    title: str
    description: str
    property_type: Optional[str] = None
    address: Optional[str] = None
    Location: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    postal_code: Optional[str] = None
    price_per_night: float
    bedrooms: Optional[int] = None
    bathrooms: Optional[int] = None
    max_guests: Optional[int] = None
    amenities: Optional[List[str]] = Field(default_factory=list)
    total_area: Optional[float] = None

class WishlistRequest(BaseModel):
    wishlist: List[WishlistItem]

@app.post("/predict")
def predict(req: PredictRequest):
    """Predicts the price of a property based on its features."""
    # (This endpoint remains unchanged)
    X = pd.DataFrame([{
        "Total_Area": req.total_area,
        "BHK": req.bhk,
        "Baths": req.baths,
        "BalconyFlag": 1 if req.balcony else 0,
        "Location": req.location
    }])
    y_log = price_pipe.predict(X)[0]
    y = np.expm1(y_log)
    return {
        "price_inr": float(y),
        "price_readable": f"₹{round(y/1e7,2)} Cr" if y >= 1e7 else f"₹{round(y/1e5,2)} Lakh"
    }


@app.post("/recommend")
def recommend(request: WishlistRequest):
    if preprocessor is None or df is None or feature_matrix is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please run the training script first.")
    
    if not request.wishlist:
        raise HTTPException(status_code=400, detail="Wishlist is empty.")
    
    wishlist_list_of_dicts = [item.model_dump() for item in request.wishlist]
    wishlist_df = pd.DataFrame(wishlist_list_of_dicts)

    
    recommended_properties = get_recommendations(request.wishlist, df, preprocessor, feature_matrix)
    enriched_properties = enrich_properties(recommended_properties)
    

    return {"properties": enriched_properties}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)


