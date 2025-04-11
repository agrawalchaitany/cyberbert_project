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
from src.utils.hardware_utils import get_hardware_info, optimize_memory, print_hardware_summary, get_optimized_settings
import argparse
from collections import Counter
from torch import nn
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
                        help='Training batch size (will be auto-adjusted based on hardware)')
    parser.add_argument('--max-length', type=int, default=256,
                        help='Maximum sequence length')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--sample-frac', type=float, default=1.0,
                        help='Fraction of data to use (for faster development)')
    parser.add_argument('--feature-count', type=int, default=30,
                        help='Number of features to select')
    parser.add_argument('--no-feature-selection', action='store_true',
                        help='Disable feature selection')
    parser.add_argument('--mixed-precision', action='store_true',
                        help='Enable mixed precision training (if supported)')
    parser.add_argument('--cache-tokenization', action='store_true',
                        help='Cache tokenized data for faster training (uses more memory)')
    parser.add_argument('--early-stopping', type=int, default=3,
                        help='Early stopping patience (epochs)')
    parser.add_argument('--eval-steps', type=int, default=100,
                        help='Evaluation steps (0 to disable)')
    return parser.parse_args()

def create_model_with_fresh_head(model_path: str, num_labels: int, hw_info: dict) -> BertForSequenceClassification:
    """Create a model with a fresh classification head for cybersecurity data"""
    try:
        print(f"Loading base model from {model_path}")
        
        # Load only the base BERT model
        base_model = BertModel.from_pretrained(model_path)
        
        # Create new classification model with optimized config
        config = BertConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            hidden_dropout_prob=0.2,  # Increase dropout for better generalization
            classifier_dropout=0.2,
            id2label={i: label for i, label in enumerate(sorted(set(labels)))},
            label2id={label: i for i, label in enumerate(sorted(set(labels)))}
        )
        
        # Add training metadata to config
        config.task_specific_params = {
            "cybersecurity_classification": {
                "num_labels": num_labels,
                "created_with": "CyberBERT Trainer",
                "training_data": os.path.basename(args.data),
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        model = BertForSequenceClassification(config)
        
        # Copy base model weights
        model.bert = base_model
        
        # Initialize the classification head weights properly
        model.classifier.weight.data.normal_(mean=0.0, std=0.02)
        model.classifier.bias.data.zero_()
        
        # Apply hardware optimizations
        optimize_memory(hw_info, model)
        
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        raise

def main():
    global args, labels
    args = parse_args()
    
    # Get hardware configuration first
    hw_info = get_hardware_info()
    print_hardware_summary(hw_info)
    
    # Get optimized settings based on hardware
    opt_settings = get_optimized_settings(hw_info)
    
    # Apply memory optimizations
    optimize_memory(hw_info)
    
    # Configuration from arguments with hardware-aware optimizations
    TRAIN_BATCH_SIZE = min(args.batch_size, hw_info['batch_size'])
    if TRAIN_BATCH_SIZE != args.batch_size:
        print(f"Warning: Reducing batch size to {TRAIN_BATCH_SIZE} for memory optimization")
    
    # Configuration from arguments
    MODEL_PATH = args.model
    DATA_PATH = args.data
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    MAX_LENGTH = min(args.max_length, opt_settings['max_length'])
    SAVE_DIR = args.output
    FEATURE_SELECTION = not args.no_feature_selection
    FEATURE_COUNT = args.feature_count
    SAMPLE_FRAC = args.sample_frac
    MIXED_PRECISION = args.mixed_precision or opt_settings['mixed_precision']
    CACHE_TOKENIZATION = args.cache_tokenization
    EARLY_STOPPING = args.early_stopping
    EVAL_STEPS = args.eval_steps

    print("\nTraining Configuration:")
    print(f"- Device: {hw_info['device']}")
    print(f"- Batch size: {TRAIN_BATCH_SIZE}")
    print(f"- Epochs: {EPOCHS}")
    print(f"- Learning rate: {LEARNING_RATE}")
    print(f"- Max sequence length: {MAX_LENGTH}")
    print(f"- Sample fraction: {SAMPLE_FRAC}")
    print(f"- Feature selection: {FEATURE_SELECTION} (features: {FEATURE_COUNT})")
    print(f"- Mixed precision: {MIXED_PRECISION}")
    print(f"- Cache tokenization: {CACHE_TOKENIZATION}")
    print(f"- Early stopping patience: {EARLY_STOPPING}")
    print(f"- Evaluation steps: {EVAL_STEPS}")
    
    # Track execution time
    start_time = time.time()
    
    # Load and prepare data with optimized feature selection
    data_loader = CyberDataLoader()
    texts, labels = data_loader.load_data(
        data_path=DATA_PATH,
        feature_selection=FEATURE_SELECTION,
        n_features=FEATURE_COUNT,
        sample_fraction=SAMPLE_FRAC
    )
    num_labels = data_loader.get_num_labels()
    
    # Create tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    
    # Create dataset with improved tokenization caching
    full_dataset = CyberSecurityDataset(
        texts, 
        labels, 
        tokenizer, 
        MAX_LENGTH, 
        cache_tokenization=CACHE_TOKENIZATION
    )
    
    # Get class weights directly from the dataset
    weight_tensor = full_dataset.get_class_weights().to(hw_info['device'])
    
    # Create model with fresh classification head
    model = create_model_with_fresh_head(MODEL_PATH, num_labels, hw_info)
    
    # Store class weights in the model config as a list (JSON serializable)
    model.config.class_weights = weight_tensor.cpu().tolist()
    
    # Verify the classifier layer shape
    print("\nModel classifier layer shape:")
    print(f"Weight shape: {model.classifier.weight.shape}")
    print(f"Bias shape: {model.classifier.bias.shape}")
    
    # Split dataset with proper stratification for imbalanced data
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Create dataloaders with optimized settings
    # For PyTorch DataLoader
    dataloader_kwargs = {
        'batch_size': TRAIN_BATCH_SIZE,
        'num_workers': opt_settings['num_workers'],
        'pin_memory': opt_settings['pin_memory'],
    }
    
    # Only add drop_last if we have enough data
    if len(full_dataset) > TRAIN_BATCH_SIZE * 10:
        dataloader_kwargs['drop_last'] = True
        
    # Create train/val split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {train_size} samples, validating on {val_size} samples")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_dataloader = DataLoader(val_dataset, **dataloader_kwargs)
    
    # Setup optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': opt_settings['weight_decay'],
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        }
    ]
    
    # Create optimizer and scheduler
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
    
    # Calculate total steps with potential gradient accumulation
    total_steps = len(train_dataloader) * EPOCHS // opt_settings['gradient_accumulation_steps']
    warmup_steps = int(total_steps * opt_settings['warmup_ratio'])
    
    # Create learning rate scheduler with warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Create enhanced trainer
    trainer = CyberBERTTrainer(model, tokenizer, hw_info['device'])
    
    # Train with enhanced monitoring
    training_results = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=EPOCHS,
        optimizer=optimizer,
        scheduler=scheduler,
        save_dir=SAVE_DIR,
        mixed_precision=MIXED_PRECISION,
        early_stopping_patience=EARLY_STOPPING,
        eval_steps=EVAL_STEPS
    )
    
    # Display final execution time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    
    # Save important features if feature selection was used
    if FEATURE_SELECTION and data_loader.selected_features:
        features_file = os.path.join(SAVE_DIR, 'selected_features.txt')
        with open(features_file, 'w') as f:
            for feature in data_loader.selected_features:
                f.write(f"{feature}\n")
        print(f"Saved {len(data_loader.selected_features)} selected features to {features_file}")

if __name__ == "__main__":
    main()
