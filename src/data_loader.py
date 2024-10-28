import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import joblib

def load_and_preprocess_data(filepath):

    data = pd.read_csv(filepath)
    label_encoder = LabelEncoder()
    data['bias_encoded'] = label_encoder.fit_transform(data['bias_rating'])

    # Balancing the dataset by oversampling or undersampling
    left_data = data[data['bias_rating'] == 'left']
    center_data = data[data['bias_rating'] == 'center']
    right_data = data[data['bias_rating'] == 'right']
    
    # Example: Oversample the minority classes to match the majority class size
    max_size = max(len(left_data), len(center_data), len(right_data))
    left_balanced = resample(left_data, replace=True, n_samples=max_size, random_state=42)
    center_balanced = resample(center_data, replace=True, n_samples=max_size, random_state=42)
    right_balanced = resample(right_data, replace=True, n_samples=max_size, random_state=42)

    # length of the balanced datasets
    print("Lengths: ", len(left_balanced), len(center_balanced), len(right_balanced))

    # Combine into a balanced dataset
    balanced_data = pd.concat([left_balanced, center_balanced, right_balanced])

    # Split the balanced dataset
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        balanced_data['title'], balanced_data['bias_encoded'], test_size=0.2, random_state=42
    )

    # Save the label encoder for future use
    joblib.dump(label_encoder, "label_encoder.joblib")
    
    return train_texts, test_texts, train_labels, test_labels, label_encoder
