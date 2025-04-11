import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

class CyberDataLoader:
    def __init__(self):
        self.label_column = None
        self.num_labels = None
        self.important_features = None
        self.selected_features = None

    def load_data(self, data_path, feature_selection=True, n_features=30, sample_fraction=1.0):
        """
        Load and preprocess the data from CSV file with optimizations.
        
        Args:
            data_path: Path to the CSV data file
            feature_selection: Whether to perform feature selection
            n_features: Number of top features to select if feature_selection is True
            sample_fraction: Fraction of data to sample for faster development
        """
        print(f"Loading data from {data_path}")
        
        # Read data
        df = pd.read_csv(data_path)
        
        if sample_fraction < 1.0:
            # Sample data for faster development cycles
            original_size = len(df)
            df = df.sample(frac=sample_fraction, random_state=42)
            print(f"Sampled {len(df)} records from {original_size} (fraction: {sample_fraction})")
        
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
        
        # Print label distribution
        print("\nLabel distribution:")
        label_counts = pd.Series(labels).value_counts()
        for label, count in label_counts.items():
            print(f"- {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # Select features
        X = df.drop(self.label_column, axis=1)
        
        # Perform feature selection if requested
        if feature_selection:
            # Keep track of feature names
            feature_names = X.columns
            
            # Handle missing or non-numeric values
            X_clean = X.fillna(0)
            
            # Select top features using ANOVA F-statistic
            selector = SelectKBest(f_classif, k=min(n_features, len(X.columns)))
            selector.fit(X_clean, labels)
            
            # Get mask of selected features
            feature_mask = selector.get_support()
            
            # Get selected feature names
            self.selected_features = feature_names[feature_mask].tolist()
            
            print(f"\nSelected top {len(self.selected_features)} features:")
            # Print top 10 features with their scores
            scores = selector.scores_
            sorted_idx = np.argsort(scores[feature_mask])[::-1]
            top_features = [self.selected_features[i] for i in sorted_idx[:10]]
            print(", ".join(top_features))
            
            # Use only selected features
            X = X[self.selected_features]
        
        # Convert numeric data to manageable precision
        for col in X.select_dtypes(include=['float64']).columns:
            X[col] = X[col].round(3)
        
        # Convert features to string for BERT processing
        features = X.values
        texts = [' '.join([f"{col}={val}" for col, val in zip(X.columns, row)]) for row in features]
        
        # Print a sample to show format
        print(f"\nSample text input for BERT:")
        print(texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0])
        
        # Store number of unique labels
        self.num_labels = len(np.unique(labels))
        print(f"\nNumber of unique labels: {self.num_labels}")
        
        return texts, labels

    def get_num_labels(self):
        """Return the number of unique labels in the dataset."""
        if self.num_labels is None:
            raise ValueError("Data hasn't been loaded yet. Call load_data() first.")
        return self.num_labels
        
    def get_selected_features(self):
        """Return the list of selected features if feature selection was performed."""
        return self.selected_features
