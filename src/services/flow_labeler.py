import torch
from transformers import BertForSequenceClassification
import numpy as np
from typing import Dict, Any

class FlowLabeler:
    """Service for real-time flow labeling using CyberBERT"""
    
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Complete label mapping for CICIDS2017 dataset
        self.label_map = {
            0: "BENIGN",
            1: "DDoS",
            2: "PortScan", 
            3: "FTP-Patator",
            4: "SSH-Patator",
            5: "DoS slowloris",
            6: "DoS Slowhttptest",
            7: "DoS GoldenEye"
            
        }
    
    def _features_to_text(self, features: Dict[str, Any]) -> str:
        """Convert numerical features to text format for BERT"""
        text_parts = []
        for key, value in features.items():
            # Skip non-numeric or irrelevant fields
            if key in ['Flow ID', 'Src IP', 'Dst IP', 'Protocol', 'Timestamp', 'Label']:
                continue
            text_parts.append(f"{key} is {value}")
        return " ".join(text_parts)
    
    @torch.no_grad()
    def predict(self, features: Dict[str, Any]) -> str:
        """Predict label for a single flow"""
        # Convert features to text
        text = self._features_to_text(features)
        
        # Tokenize
        inputs = self.model.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get prediction
        outputs = self.model(**inputs)
        prediction = outputs.logits.argmax(-1).item()
        
        return self.label_map.get(prediction, "Unknown")
