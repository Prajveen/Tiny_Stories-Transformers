from fastapi import FastAPI
import torch
from pydantic import BaseModel
from app.model.inference import *
import numpy as np

# Instantiate FastAPI app
app = FastAPI()


class PredictionOut(BaseModel):
    story: str

class PredictionIn(BaseModel):
    start_word: str

@app.get("/")
def home():
    return {"health_check": "OK"}

# Define inference endpoint
@app.post("/predict/", response_model=PredictionOut)
def predict(start_word: str):
    print(f"start_word:{start_word}")
    
    result = PredictionOut(story=generate_text(start_word))
    
    return result

# Run the FastAPI app using Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)