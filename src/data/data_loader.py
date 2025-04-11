import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import SelectKBest, f_classif

class CyberDataLoader:
    """
    Simplified data loader for CyberBERT with streamlined functionality
    """
    def __init__(self):
        self.label_column = None
        self.num_labels = None
        self.selected_features = None
        # Define the expected labels in the correct order according to the model's label mapping
        self.expected_labels = [
            "BENIGN",
            "DoS GoldenEye",
            "DoS Slowhttptest",
            "FTP-Patator",
            "PortScan",
            "SSH-Patator"
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

    def _preprocess_data(self, df):
        """
        Preprocess data to handle problematic values:
        - Replace infinities with NaN
        - Replace NaN with 0
        - Cap very large values to prevent float64 overflow
        - Add small random noise to constant features
        """
        print("Preprocessing data to handle infinities and extreme values")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Replace infinities with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # Get numeric columns only
        numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
        
        # Calculate reasonable caps based on data distribution
        caps = {}
        for col in numeric_cols:
            # Get percentiles, ignoring NaN values
            values = df_clean[col].dropna()
            if len(values) > 0:
                q99 = values.quantile(0.99)
                q01 = values.quantile(0.01)
                
                # Set caps at 3x the 99th percentile or 1e9, whichever is smaller
                upper_cap = min(q99 * 3, 1e9)
                lower_cap = max(q01 * 3 * (-1), -1e9)
                
                caps[col] = (lower_cap, upper_cap)
        
        # Cap extremely large values
        capped_count = 0
        for col in numeric_cols:
            if col in caps:
                lower_cap, upper_cap = caps[col]
                # Count values outside cap range
                count_before = df_clean[col].isna().sum()
                
                # Cap values
                df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
                
                # Replace any new NaNs from the clipping operation
                count_after = df_clean[col].isna().sum()
                capped_count += (count_after - count_before)
        
        if capped_count > 0:
            print(f"Capped {capped_count} extreme values across all features")
        
        # Finally replace remaining NaNs with 0
        df_clean = df_clean.fillna(0)
        
        # Convert numeric data to manageable precision
        for col in df_clean.select_dtypes(include=['float64']).columns:
            df_clean[col] = df_clean[col].round(6)
        
        return df_clean

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
        
        # Preprocess data - handle missing values, infinities, and extreme values
        X = df.drop(self.label_column, axis=1)
        X = self._preprocess_data(X)
        
        # Check for any remaining problematic values after preprocessing
        problematic = np.any(np.isnan(X.values)) or np.any(np.isinf(X.values))
        if problematic:
            print("Warning: There are still NaN or infinity values after preprocessing")
        
        # Perform feature selection if requested
        if feature_selection:
            X = self._select_features(X, labels, n_features)
        
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
        
        try:
            # Check for constant features before selection
            variance = X.var()
            constant_features = variance[variance <= 1e-10].index.tolist()
            
            if constant_features:
                print(f"Detected {len(constant_features)} constant or near-constant features")
                
                # Add tiny random noise to constant features to prevent warnings
                for col in constant_features:
                    X[col] = X[col] + np.random.normal(0, 1e-6, size=len(X))
            
            # Temporarily suppress warnings during feature selection
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                warnings.simplefilter("ignore", category=RuntimeWarning)
                
                # Select top features
                selector = SelectKBest(f_classif, k=min(n_features, len(X.columns)))
                selector.fit(X, labels)
            
            # Get mask of selected features
            feature_mask = selector.get_support()
            
            # Get selected feature names
            self.selected_features = feature_names[feature_mask].tolist()
            
            print(f"\nSelected top {len(self.selected_features)} features:")
            # Print top 10 features with their scores
            scores = selector.scores_
            # Replace NaN scores with 0 and Inf with a large value
            scores = np.nan_to_num(scores, nan=0.0, posinf=1e10, neginf=-1e10)
            
            # Get indices of top scored features
            sorted_idx = np.argsort(scores[feature_mask])[::-1]
            top_features = [self.selected_features[i] for i in sorted_idx[:10]]
            print(", ".join(top_features))
            
            # Return only selected features
            return X[self.selected_features]
        except Exception as e:
            print(f"Error during feature selection: {str(e)}")
            print("Falling back to using all features")
            self.selected_features = feature_names.tolist()
            return X

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
