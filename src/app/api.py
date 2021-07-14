import uvicorn
from typing import Dict
from fastapi import FastAPI, Depends
from pydantic import BaseModel
from datetime import datetime

from src.models.model import SentimentModel, get_model

app = FastAPI()

class SentimentRequest(BaseModel):
    text: str
    
class SentimentResponse(BaseModel):
    probability: Dict[str, float]
    sentiment: str
    confidence: float
    
@app.get("/")
def read_root():
    return {"main": "API Server " + datetime.now().strftime("%Y%m%d %H:%M:%S")}

@app.post("/predict", response_model=SentimentResponse)
def predict(request: SentimentRequest, model: SentimentModel = Depends(get_model)):
    sentiment, confidence, probability = model.predict(request.text)
    return SentimentResponse(sentiment=sentiment, confidence=confidence, probability=probability)

if __name__ == "__main__":
    uvicorn(app, host="127.0.0.1", port=8080, reload=True)
