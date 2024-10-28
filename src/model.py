from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

def initialize_model(num_labels):
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    return model, tokenizer

def save_model(model, tokenizer, directory="./bias_classifier_model"):
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)

def load_model(directory="./bias_classifier_model"):
    model = DistilBertForSequenceClassification.from_pretrained(directory)
    tokenizer = DistilBertTokenizer.from_pretrained(directory)
    return model, tokenizer
