import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

class CyberDataLoader:
    """
    Simplified data loader for CyberBERT with streamlined functionality
    """
    def __init__(self):
        self.label_column = None
        self.num_labels = None
        self.selected_features = None
        # Define the expected labels
        self.expected_labels = [
            "BENIGN",
            "DDoS",
            "PortScan", 
            "FTP-Patator",
            "SSH-Patator",
            "DoS slowloris",
            "DoS Slowhttptest",
            "DoS GoldenEye"
        ]
        
        # Label normalization mapping
        self.label_mapping = {
            "benign": "BENIGN",
            "normal": "BENIGN",
            "ddos": "DDoS",
            "portscan": "PortScan",
            "port-scan": "PortScan",
            "port_scan": "PortScan",
            "ftp-patator": "FTP-Patator",
            "ftp_patator": "FTP-Patator",
            "ftppatator": "FTP-Patator",
            "ssh-patator": "SSH-Patator",
            "ssh_patator": "SSH-Patator",
            "sshpatator": "SSH-Patator",
            "dos slowloris": "DoS slowloris",
            "dos-slowloris": "DoS slowloris",
            "dos_slowloris": "DoS slowloris",
            "slowloris": "DoS slowloris",
            "dos slowhttptest": "DoS Slowhttptest",
            "dos-slowhttptest": "DoS Slowhttptest",
            "dos_slowhttptest": "DoS Slowhttptest",
            "slowhttptest": "DoS Slowhttptest",
            "dos goldeneye": "DoS GoldenEye",
            "dos-goldeneye": "DoS GoldenEye",
            "dos_goldeneye": "DoS GoldenEye",
            "goldeneye": "DoS GoldenEye"
        }

    def normalize_labels(self, labels):
        """Normalize label names to ensure consistency with the expected labels"""
        normalized = []
        
        for label in labels:
            # Convert to string in case it's not already
            label_str = str(label).strip()
            # Check if label needs normalization
            if label_str.lower() in self.label_mapping:
                normalized.append(self.label_mapping[label_str.lower()])
            else:
                # Keep the original if it's already in expected format
                if label_str in self.expected_labels:
                    normalized.append(label_str)
                else:
                    print(f"Warning: Unknown label '{label_str}', treating as 'BENIGN'")
                    normalized.append("BENIGN")
        
        return np.array(normalized)

    def load_data(self, data_path, feature_selection=True, n_features=30, sample_fraction=1.0):
        """
        Load and preprocess the data from CSV file
        
        Args:
            data_path: Path to the CSV data file
            feature_selection: Whether to perform feature selection
            n_features: Number of top features to select if feature_selection is True
            sample_fraction: Fraction of data to sample for faster development
        
        Returns:
            texts: List of feature strings formatted for BERT
            labels: Array of normalized label strings
        """
        print(f"Loading data from {data_path}")
        
        # Read data
        df = pd.read_csv(data_path)
        
        if sample_fraction < 1.0:
            # Sample data for faster development cycles
            original_size = len(df)
            df = df.sample(frac=sample_fraction, random_state=42)
            print(f"Sampled {len(df)} records from {original_size} (fraction: {sample_fraction})")
        
        # Find label column
        possible_label_columns = ['Label', 'label', 'class', 'Class', 'target', 'Target']
        for col in possible_label_columns:
            if col in df.columns:
                self.label_column = col
                break
        
        if self.label_column is None:
            raise ValueError(f"No label column found. Expected one of: {possible_label_columns}")

        # Extract and normalize labels
        labels = self.normalize_labels(df[self.label_column].values)
        
        # Print label distribution
        print("\nLabel distribution:")
        for label, count in pd.Series(labels).value_counts().items():
            print(f"- {label}: {count} samples ({count/len(labels)*100:.1f}%)")
        
        # Select features
        X = df.drop(self.label_column, axis=1)
        
        # Perform feature selection if requested
        if feature_selection:
            X = self._select_features(X, labels, n_features)
        
        # Convert numeric data to manageable precision
        for col in X.select_dtypes(include=['float64']).columns:
            X[col] = X[col].round(3)
        
        # Convert features to string for BERT processing
        texts = [' '.join([f"{col}={val}" for col, val in zip(X.columns, row)]) for row in X.values]
        
        # Print a sample
        print(f"\nSample text input for BERT:")
        print(texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0])
        
        # Store number of unique labels
        self.num_labels = len(np.unique(labels))
        print(f"\nNumber of unique labels: {self.num_labels}")
        
        return texts, labels
    
    def _select_features(self, X, labels, n_features):
        """Select the most important features using ANOVA F-statistic"""
        # Keep track of feature names
        feature_names = X.columns
        
        # Handle missing values
        X_clean = X.fillna(0)
        
        # Select top features
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
        
        # Return only selected features
        return X[self.selected_features]

    def get_num_labels(self):
        """Return the number of unique labels in the dataset"""
        if self.num_labels is None:
            raise ValueError("Data hasn't been loaded yet. Call load_data() first")
        return self.num_labels
        
    def get_selected_features(self):
        """Return the list of selected features if feature selection was performed"""
        return self.selected_features
        
    def get_expected_labels(self):
        """Return the list of expected labels in the correct order"""
        return self.expected_labels[:self.num_labels] if self.num_labels else self.expected_labels
