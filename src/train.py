from transformers import Trainer, TrainingArguments
from sklearn.utils.class_weight import compute_class_weight
import torch
from src.dataset import BiasDataset
import numpy as np

# Calculate class weights based on the training labels
def get_class_weights(train_labels):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1, 2]), y=train_labels)
    return torch.tensor(class_weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Ensure class weights are on the same device as the model
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Move inputs to the device
        labels = inputs.pop("labels").to(self.args.device)
        inputs = {key: val.to(self.args.device) for key, val in inputs.items()}

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute loss with class weights
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model(model, train_encodings, train_labels, test_encodings, test_labels):
    # Prepare the datasets
    train_dataset = BiasDataset(train_encodings, train_labels)
    test_dataset = BiasDataset(test_encodings, test_labels)
    
    # Calculate class weights
    class_weights = get_class_weights(train_labels)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=5,            # Increase to more epochs
        learning_rate=2e-5,            # Reduce learning rate
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch"
    )

    # Initialize WeightedTrainer with class weights
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        class_weights=class_weights
    )

    # Start training
    trainer.train()
    return trainer
