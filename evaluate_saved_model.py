import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from sklearn.metrics import classification_report
import joblib
from src.dataset import BiasDataset
from src.data_loader import load_and_preprocess_data

# Load the saved model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./bias_classifier_model", resume_download=True)
tokenizer = DistilBertTokenizer.from_pretrained("./bias_classifier_model", resume_download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Move model to device
model.to(device)

# Load the label encoder
label_encoder = joblib.load("label_encoder.joblib")

# Load the dataset
train_texts, test_texts, train_labels, test_labels, _ = load_and_preprocess_data(
    'data/allsides_balanced_news_headlines-texts.csv'
)

# Tokenize the test texts
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)

# Prepare the test dataset
test_dataset = BiasDataset(test_encodings, test_labels)

def evaluate_saved_model(model, test_dataset, test_labels, label_encoder):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    
    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for i in range(len(test_dataset)):
            # Convert each item in test_dataset.encodings to tensor
            print(i)
            inputs = {key: torch.tensor([val[i]]).to(model.device) for key, val in test_dataset.encodings.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            predictions.append(predicted_class_id)
    
    # Map predictions to labels
    predicted_labels = [label_encoder.inverse_transform([pred])[0] for pred in predictions]
    true_labels = [label_encoder.inverse_transform([label])[0] for label in test_labels]
    
    # Print the classification report
    print(classification_report(true_labels, predicted_labels, target_names=label_encoder.classes_))

# Run the evaluation
evaluate_saved_model(model, test_dataset, test_labels, label_encoder)

