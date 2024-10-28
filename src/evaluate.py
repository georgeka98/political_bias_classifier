from sklearn.metrics import classification_report
import torch

def evaluate_model(trainer, test_dataset, test_labels, label_encoder):
    # Make predictions on the test dataset
    predictions = trainer.predict(test_dataset)
    preds = torch.argmax(torch.tensor(predictions.predictions), dim=1).numpy()
    
    # Print classification report to assess accuracy, precision, recall, F1 score
    print(classification_report(test_labels, preds, target_names=label_encoder.classes_))
