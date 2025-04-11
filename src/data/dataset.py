import torch
from torch.utils.data import Dataset
import numpy as np

class CyberSecurityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, cache_tokenization=True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization
        
        # Create label encoder
        self.unique_labels = sorted(set(labels))
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        # Convert string labels to numeric indices
        self.labels = [self.label_to_id[label] for label in labels]
        
        # Pre-tokenize if cache_tokenization is enabled (faster but uses more memory)
        self.encodings = None
        if self.cache_tokenization:
            self.encodings = self._tokenize_all_texts()
        
    def _tokenize_all_texts(self):
        """Pre-tokenize all texts for faster training"""
        return self.tokenizer(
            self.texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        label = self.labels[idx]
        
        # Use cached tokenization if available
        if self.encodings is not None:
            return {
                'input_ids': self.encodings['input_ids'][idx],
                'attention_mask': self.encodings['attention_mask'][idx],
                'label': torch.tensor(label, dtype=torch.long)
            }
        
        # Otherwise tokenize on-the-fly
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def get_num_labels(self):
        return len(self.unique_labels)
        
    def get_label_map(self):
        return self.label_to_id.copy()
        
    def get_class_weights(self):
        """Compute class weights for imbalanced datasets"""
        # Count examples per class
        class_counts = np.bincount(self.labels)
        # Create weights that are inversely proportional to class frequencies
        n_samples = len(self.labels)
        n_classes = len(class_counts)
        weights = n_samples / (n_classes * class_counts)
        return torch.tensor(weights, dtype=torch.float)
