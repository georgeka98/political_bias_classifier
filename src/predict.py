import torch
import joblib

def predict_bias(text, model, tokenizer):
    # Load the label encoder
    label_encoder = joblib.load("label_encoder.joblib")
    
    # Preprocess and predict
    inputs = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Convert prediction to label
    return label_encoder.inverse_transform([predicted_class_id])[0]
