import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import json
import gc

class CyberSecurityDataset(Dataset):
    """
    A simplified PyTorch Dataset for CyberBERT that handles tokenization and caching
    """
    def __init__(self, texts, labels, tokenizer_name_or_path="bert-base-uncased", 
                 max_length=128, cache_dir="cache"):
        """
        Initialize the dataset with texts and labels
        
        Args:
            texts: List of text samples (feature strings)
            labels: Array of label strings
            tokenizer_name_or_path: BERT tokenizer name or path
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to store tokenization cache
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
        
        # Create unique labels mapping (label string -> int)
        self.unique_labels = sorted(list(set(labels)))
        self.label_map = {label: i for i, label in enumerate(self.unique_labels)}
        
        # Print label mapping
        print("Label mapping:")
        for label, idx in self.label_map.items():
            print(f"  {label} -> {idx}")
        
        # Enable caching to speed up repeated runs
        self._setup_cache()
        
    def _setup_cache(self):
        """Set up tokenization cache system"""
        # Create cache directory if it doesn't exist
        self.use_cache = self.cache_dir is not None
        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Create a cache filename based on tokenizer and data
            tokenizer_name = self.tokenizer.name_or_path.replace('/', '_')
            data_hash = hash(tuple(self.texts[:10]))  # Hash a sample of texts for filename
            self.cache_file = os.path.join(
                self.cache_dir, 
                f"tokenized_cache_{tokenizer_name}_{data_hash}_{self.max_length}.json"
            )
            
            # Try to load cached tokenization
            self.tokenized_texts = self._load_cache()
            
            if self.tokenized_texts is None:
                # Cache miss - tokenize and save
                self.tokenized_texts = self._tokenize_all_texts()
                self._save_cache()
                print(f"Tokenized texts saved to cache: {self.cache_file}")
            else:
                print(f"Loaded tokenized texts from cache: {self.cache_file}")
        else:
            # No caching - tokenize directly
            self.tokenized_texts = self._tokenize_all_texts()

        # After tokenization, we can potentially free memory by releasing the original texts
        # if we have successfully tokenized everything
        if len(self.tokenized_texts) == len(self.texts):
            # Keep a small reference to length for validation
            self._texts_len = len(self.texts)
            self.texts = None
            gc.collect() # Force garbage collection
            
    def _tokenize_all_texts(self):
        """Tokenize all texts in the dataset with memory optimization"""
        print(f"Tokenizing {len(self.texts)} texts (max_length={self.max_length})...")
        
        # Process in batches to reduce memory usage
        tokenized = []
        batch_size = 1000  # Process 1000 texts at a time
        
        for start_idx in range(0, len(self.texts), batch_size):
            end_idx = min(start_idx + batch_size, len(self.texts))
            batch_texts = self.texts[start_idx:end_idx]
            
            print(f"  Tokenized {start_idx}/{len(self.texts)} texts")
            
            # Batch tokenization
            encoded_batch = self.tokenizer(
                batch_texts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors=None  # Return as python lists
            )
            
            # Convert from batch format to individual samples
            for i in range(len(batch_texts)):
                sample = {
                    'input_ids': encoded_batch['input_ids'][i],
                    'attention_mask': encoded_batch['attention_mask'][i]
                }
                tokenized.append(sample)
                
            # Free memory after each batch
            del batch_texts, encoded_batch
            gc.collect()
            
        print(f"Tokenization complete for {len(tokenized)} texts")
        return tokenized
        
    def _load_cache(self):
        """Load tokenized texts from cache file if it exists"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                
                # Verify cache is valid (check a sample)
                if len(cache_data) == len(self.texts):
                    print(f"Cache hit: Loading {len(cache_data)} tokenized texts")
                    return cache_data
            except Exception as e:
                print(f"Cache loading failed: {e}")
        
        print("Cache miss: Will tokenize texts")
        return None
        
    def _save_cache(self):
        """Save tokenized texts to cache file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.tokenized_texts, f)
        except Exception as e:
            print(f"Failed to save cache: {e}")
            
    def __len__(self):
        """Return dataset size"""
        return len(self.tokenized_texts)
        
    def __getitem__(self, idx):
        """Get tokenized text and label for an index, optimized for memory"""
        # Get tokenized text
        encoded = self.tokenized_texts[idx]
        
        # Convert to tensors
        item = {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
        }
        
        # Convert label string to index
        label_idx = self.label_map[self.labels[idx]]
        item['labels'] = torch.tensor(label_idx, dtype=torch.long)
        
        return item
        
    def get_label_mapping(self):
        """Return the mapping of label string to index"""
        return self.label_map
        
    def get_num_labels(self):
        """Return the number of unique labels"""
        return len(self.unique_labels)
        
    def get_class_weights(self):
        """
        Calculate class weights inversely proportional to class frequencies
        for handling imbalanced datasets
        """
        # Count instances per class
        label_indices = [self.label_map[label] for label in self.labels]
        class_counts = np.bincount(label_indices, minlength=len(self.unique_labels))
        
        # Calculate weights (inversely proportional to frequency)
        total = len(self.labels)
        class_weights = total / (class_counts * len(self.unique_labels))
        
        # Convert to tensor
        weights = torch.tensor(class_weights, dtype=torch.float)
        
        print("Class weights to handle imbalance:")
        for label, weight in zip(self.unique_labels, class_weights):
            print(f"  {label}: {weight:.4f}")
            
        return weights
