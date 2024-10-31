from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import joblib
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model, tokenizer, and label encoder
model = DistilBertForSequenceClassification.from_pretrained("./bias_classifier_model", resume_download=True)
tokenizer = DistilBertTokenizer.from_pretrained("./bias_classifier_model", resume_download=True)
label_encoder = joblib.load("label_encoder.joblib")  # Ensure this file is saved during training

# Define the request schema with Pydantic
class PredictRequest(BaseModel):
    text: str

# Serve static files from the "static" directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the index.html file at the root URL
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join("static", "index.html"))

# Define the /predict endpoint
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Tokenize the input text
        inputs = tokenizer(request.text, truncation=True, padding=True, max_length=128, return_tensors="pt")

        print(inputs)
        
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
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"We need to raise the minimum wage to support working families.\"}"
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Capitalism promotes individual responsibility and opportunity.\"}"
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Government regulation should be minimized to allow businesses to thrive.\"}"
# curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"text\": \"Corporations must be held accountable for their environmental impact.\"}"

# Left-Leaning Phrases
# Economic and Social Welfare

# "We need to raise the minimum wage to support working families."
# "Universal healthcare should be a fundamental right, accessible to everyone."
# "Increasing taxes on the wealthiest individuals is essential for a fair economy."
# "Investing in public education is the best way to ensure equal opportunities for all children."
# "Corporations must be held accountable for their environmental impact."
# Climate Change and Environment

# "We need immediate action on climate change to protect future generations."
# "Renewable energy sources like wind and solar are crucial for a sustainable future."
# "The government should enforce stricter regulations on carbon emissions."
# "The Green New Deal offers a comprehensive plan to combat climate change."
# "Protecting biodiversity and endangered species is a moral responsibility."
# Social Justice and Equality

# "Police reform is essential to address systemic issues within law enforcement."
# "The government should protect the rights of marginalized communities."
# "Affordable housing is a human right that everyone should have access to."
# "We need to close the gender pay gap to promote equality in the workplace."
# "Immigration policies should prioritize human dignity and compassion."
# Right-Leaning Phrases
# Economic Policies and Taxes

# "Lowering taxes will stimulate job creation and economic growth."
# "Government regulation should be minimized to allow businesses to thrive."
# "The free market is the best way to allocate resources and drive innovation."
# "We must prioritize reducing the national debt to secure our economic future."
# "Capitalism promotes individual responsibility and opportunity."
# National Security and Law Enforcement

# "A strong military is essential for maintaining peace and security."
# "Border security is crucial to maintain national sovereignty."
# "The Second Amendment protects our right to defend ourselves and our families."
# "We need stricter immigration policies to protect American jobs and culture."
# "Supporting law enforcement is vital to keeping our communities safe."
# Traditional Values and Social Policies

# "Family values and traditional institutions are the foundation of society."
# "Religious freedom must be protected against government overreach."
# "Parents should have more control over their children’s education."
# "Marriage should be between a man and a woman."
# "The role of government should be limited in individual lives."
# Centrist Phrases
# Balanced Economic and Social Policies

# "Fiscal responsibility and social welfare can go hand-in-hand."
# "Balancing environmental protection with economic growth is essential."
# "We need a bipartisan approach to healthcare reform."
# "Both public and private sectors play important roles in the economy."
# "Moderate tax policies can ensure fairness without stifling growth."
# Pragmatic Social Policies

# "Gun laws should protect citizens while respecting the Second Amendment."
# "Immigration policies should focus on security and opportunity."
# "We should address climate change while supporting energy independence."
# "Law enforcement reform should consider both community safety and justice."
# "Education reform should focus on both funding and accountability."
# Unity and Bipartisanship

# "Working across the aisle is essential to solving major issues."
# "Respect for diverse opinions is crucial to a healthy democracy."
# "We need leaders who prioritize unity over party politics."
# "Solutions should come from both liberal and conservative ideas."
# "Effective governance requires compromise and cooperation."




# uvicorn api:app --reload