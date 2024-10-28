from src.data_loader import load_and_preprocess_data
from src.model import initialize_model, save_model
from src.train import train_model
from src.evaluate import evaluate_model
from transformers import DistilBertTokenizer
from src.predict import predict_bias
from src.dataset import BiasDataset
import joblib
import torch

# Load data and preprocess
train_texts, test_texts, train_labels, test_labels, label_encoder = load_and_preprocess_data(
    'data/allsides_balanced_news_headlines-texts.csv'
)



# Initialize model and tokenizer
model, tokenizer = initialize_model(num_labels=len(label_encoder.classes_))

# Set up device to use MPS if available, otherwise fall back to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move model to device
model.to(device)

# Tokenize data
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Train the model
trainer = train_model(model, train_encodings, train_labels, test_encodings, test_labels)

# Evaluate the model
test_dataset = BiasDataset(test_encodings, test_labels)
evaluate_model(trainer, test_dataset, test_labels, label_encoder)

# Save the model and tokenizer
save_model(model, tokenizer)
