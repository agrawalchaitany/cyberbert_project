import torch
from torch.utils.data import Dataset

class CyberSecurityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label encoder
        self.unique_labels = sorted(set(labels))
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        
        # Convert string labels to numeric indices
        self.labels = [self.label_to_id[label] for label in labels]
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
