import os
import sys
import gc
import psutil
import torch
from torch.optim import AdamW
import platform
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertModel
from src.data.dataset import CyberSecurityDataset
from src.training.trainer import CyberBERTTrainer
from src.data.data_loader import CyberDataLoader
from src.utils.hardware_utils import get_hardware_info, optimize_memory, print_hardware_summary
import argparse
from collections import Counter
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(description='Train CyberBERT model on network flow data')
    parser.add_argument('--data', type=str, default='data/processed/clean_data.csv',
                        help='Path to the input CSV data file')
    parser.add_argument('--model', type=str, default='models/cyberbert_model',
                        help='Path to pre-trained BERT model')
    parser.add_argument('--output', type=str, default='models/trained_cyberbert',
                        help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--max-length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    return parser.parse_args()

def create_model_with_fresh_head(model_path: str, num_labels: int) -> BertForSequenceClassification:
    # Load only the base BERT model
    base_model = BertModel.from_pretrained(model_path)
    
    # Create new classification model
    model = BertForSequenceClassification(
        config=BertConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            hidden_dropout_prob=0.2,  # Increase dropout for better generalization
            classifier_dropout=0.2
        )
    )
    
    # Copy base model weights
    model.bert = base_model
    
    # Initialize the classification head weights properly
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    model.classifier.bias.data.zero_()
    
    return model

def main():
    args = parse_args()
    
    # Get hardware configuration first
    hw_info = get_hardware_info()
    print_hardware_summary(hw_info)
    
    # Apply memory optimizations
    optimize_memory(hw_info)
    
    # Configuration from arguments with hardware-aware batch size
    TRAIN_BATCH_SIZE = min(args.batch_size, hw_info['batch_size'])
    if TRAIN_BATCH_SIZE != args.batch_size:
        print(f"Warning: Reducing batch size to {TRAIN_BATCH_SIZE} for memory optimization")
    
    # Configuration from arguments
    MODEL_PATH = args.model
    DATA_PATH = args.data
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = args.max_length
    SAVE_DIR = args.output

    print("\nTraining Configuration:")
    print(f"- Device: {hw_info['device']}")
    print(f"- Batch size: {TRAIN_BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Max sequence length: {MAX_LENGTH}")
    
    # Load and prepare data first to get number of classes
    data_loader = CyberDataLoader()
    texts, labels = data_loader.load_data(DATA_PATH)
    num_labels = data_loader.get_num_labels()
    
    # Print diagnostic information
    unique_labels = sorted(set(labels))
    print("\nData Analysis:")
    print(f"Number of unique labels: {num_labels}")
    print(f"Unique labels found: {unique_labels}")
    print("\nLabel distribution:")
    label_counts = Counter(labels)
    for label, count in label_counts.items():
        print(f"- {label}: {count} samples")
    
    # Calculate class weights for handling imbalance
    total_samples = sum(label_counts.values())
    class_weights = {label: total_samples / (len(label_counts) * count) 
                    for label, count in label_counts.items()}
    print("\nClass weights for handling imbalance:")
    for label, weight in class_weights.items():
        print(f"- {label}: {weight:.4f}")
    
    # Create model with fresh classification head
    model = create_model_with_fresh_head(MODEL_PATH, num_labels)
    
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    
    # Apply hardware optimizations
    optimize_memory(hw_info, model)
    
    # Create dataset with now-defined tokenizer
    full_dataset = CyberSecurityDataset(texts, labels, tokenizer, MAX_LENGTH)
    
    # Use dataset's label mapping for class weights
    label_to_id = full_dataset.get_label_map()
    weight_tensor = torch.tensor(
        [class_weights[label] for label in full_dataset.unique_labels],
        device=hw_info['device']
    )
    
    # Create loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    
    # Add criterion to model config
    model.config.loss_function = criterion
    
    # Verify the classifier layer shape
    print("\nModel classifier layer shape:")
    print(f"Weight shape: {model.classifier.weight.shape}")
    print(f"Bias shape: {model.classifier.bias.shape}")
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Create trainer with proper device from hardware info
    trainer = CyberBERTTrainer(model, tokenizer, hw_info['device'])
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=SAVE_DIR
    )

if __name__ == "__main__":
    main()
