import pandas as pd
import numpy as np

class CyberDataLoader:
    def __init__(self):
        self.label_column = None
        self.num_labels = None

    def load_data(self, data_path):
        """Load and preprocess the data from CSV file."""
        df = pd.read_csv(data_path)
        
        # Try different possible label column names
        possible_label_columns = ['Label', 'label', 'class', 'Class', 'target', 'Target']
        for col in possible_label_columns:
            if col in df.columns:
                self.label_column = col
                break
        
        if self.label_column is None:
            raise ValueError(f"No label column found. Expected one of: {possible_label_columns}")

        # Extract features and labels
        labels = df[self.label_column].values
        features = df.drop(self.label_column, axis=1).values
        
        # Convert features to string type for BERT processing
        texts = [' '.join(map(str, row)) for row in features]
        
        # Store number of unique labels
        self.num_labels = len(np.unique(labels))
        
        return texts, labels

    def get_num_labels(self):
        """Return the number of unique labels in the dataset."""
        if self.num_labels is None:
            raise ValueError("Data hasn't been loaded yet. Call load_data() first.")
        return self.num_labels
