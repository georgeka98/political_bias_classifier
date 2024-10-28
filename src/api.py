from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load the model, tokenizer, and label encoder
model = DistilBertForSequenceClassification.from_pretrained("./bias_classifier_model")
tokenizer = DistilBertTokenizer.from_pretrained("./bias_classifier_model")
label_encoder = joblib.load("label_encoder.joblib")  # Ensure this file is saved during training

# Define the request schema with Pydantic
class PredictRequest(BaseModel):
    text: str

# Define the /predict endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.text, truncation=True, padding=True, max_length=128, return_tensors="pt")
        
        # Pass inputs through the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
        
        # Map the prediction to the label
        label = label_encoder.inverse_transform([predicted_class_id])[0]
        
        # Return the prediction
        return {"prediction": label}
    
    except Exception as e:
        # Raise HTTP exception if any error occurs
        raise HTTPException(status_code=500, detail=str(e))
    

# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Iran's Revolutionary Guards seized a British-flagged oil tanker in the Strait of Hormuz last Friday in revenge...\"}"