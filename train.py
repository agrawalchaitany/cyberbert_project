#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CyberBERT Training Script - Production Ready Version
Trains a BERT-based model for network traffic classification
"""

import os
import sys
import gc
import psutil
import torch
from torch.optim import AdamW
import platform
from typing import Tuple, Dict, Any
from torch.utils.data import DataLoader, random_split
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup, BertConfig, BertModel
import argparse
from collections import Counter
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import traceback
import json

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data.dataset import CyberSecurityDataset
from src.training.trainer import CyberBERTTrainer
from src.data.data_loader import CyberDataLoader
from src.utils.hardware_utils import get_hardware_info, optimize_memory, print_hardware_summary, get_optimized_settings
from src.utils.logger import Logger
from src.utils.config import Config

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CyberBERT model on network flow data')
    
    # Configuration
    parser.add_argument('--config', type=str, help='Path to configuration file (.yaml or .json)')
    
    # Data arguments
    parser.add_argument('--data', type=str, help='Path to the input CSV data file')
    parser.add_argument('--sample-frac', type=float, help='Fraction of data to use (for faster development)')
    parser.add_argument('--feature-count', type=int, help='Number of features to select')
    parser.add_argument('--no-feature-selection', action='store_true', help='Disable feature selection')
    parser.add_argument('--cache-tokenization', action='store_true', help='Cache tokenized data for faster training')
    parser.add_argument('--max-length', type=int, help='Maximum sequence length')
    
    # Model arguments
    parser.add_argument('--model', type=str, help='Path to pre-trained BERT model')
    parser.add_argument('--output', type=str, help='Directory to save the trained model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    
    # Training arguments
    parser.add_argument('--mixed-precision', action='store_true', help='Enable mixed precision training')
    parser.add_argument('--early-stopping', type=int, help='Early stopping patience (epochs)')
    parser.add_argument('--eval-steps', type=int, help='Evaluation steps (0 to disable)')
    
    # System arguments
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    parser.add_argument('--no-log-file', action='store_true', help='Disable logging to file')
    
    return parser.parse_args()

def create_model_with_fresh_head(model_path: str, num_labels: int, hw_info: Dict[str, Any], 
                                data_loader: CyberDataLoader, logger) -> BertForSequenceClassification:
    """
    Create a model with a fresh classification head for cybersecurity data
    
    Args:
        model_path: Path to pre-trained BERT model
        num_labels: Number of classification labels
        hw_info: Hardware information
        data_loader: Data loader with label information
        logger: Logger instance
    
    Returns:
        Initialized BERT model for sequence classification
    """
    try:
        logger.info(f"Loading base model from {model_path}")
        
        # Load only the base BERT model
        base_model = BertModel.from_pretrained(model_path)
        
        # Get expected labels in the correct order
        expected_labels = data_loader.get_expected_labels()
        
        # Create label mappings for config using the expected label ordering
        id2label = {i: label for i, label in enumerate(expected_labels)}
        label2id = {label: i for i, label in enumerate(expected_labels)}
        
        logger.info(f"Label mapping configured for {num_labels} classes:")
        for idx, label in id2label.items():
            logger.debug(f"  {idx}: '{label}'")
        
        # Create new classification model with optimized config
        config = BertConfig.from_pretrained(
            model_path,
            num_labels=num_labels,
            hidden_dropout_prob=0.2,
            classifier_dropout=0.2,
            id2label=id2label,
            label2id=label2id
        )
        
        # Add training metadata to config
        config.task_specific_params = {
            "cybersecurity_classification": {
                "num_labels": num_labels,
                "created_with": "CyberBERT Trainer",
                "training_data": os.path.basename(args.data),
                "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hardware": hw_info['description'],
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
        logger.error(f"Error creating model: {str(e)}")
        logger.debug(traceback.format_exc())
        raise

def main():
    try:
        # Parse arguments
        global args
        args = parse_args()
        
        # Initialize configuration
        config = Config(args.config if args.config else None)
        config.update_from_args(args)
        
        # Initialize logger
        log_level = config.get('system.log_level', 'INFO')
        log_to_file = not args.no_log_file if hasattr(args, 'no_log_file') else config.get('system.log_to_file', True)
        logger_instance = Logger(
            name="cyberbert", 
            log_level=log_level,
            log_to_file=log_to_file
        )
        logger = logger_instance.get_logger()
        
        logger.info("CyberBERT Training - Starting")
        
        # Get hardware configuration
        hw_info = get_hardware_info()
        print_hardware_summary(hw_info)
        logger.info(f"Hardware: {hw_info['description']}")
        
        # Get optimized settings based on hardware
        opt_settings = get_optimized_settings(hw_info)
        
        # Apply memory optimizations
        optimize_memory(hw_info)
        
        # Configuration from arguments with hardware-aware optimizations
        TRAIN_BATCH_SIZE = min(config.get('model.batch_size'), hw_info['batch_size'])
        if TRAIN_BATCH_SIZE != config.get('model.batch_size'):
            logger.warning(f"Reducing batch size to {TRAIN_BATCH_SIZE} for memory optimization")
        
        # Extract configuration
        MODEL_PATH = config.get('model.model_path')
        DATA_PATH = config.get('data.data_path')
        EPOCHS = config.get('model.epochs')
        LEARNING_RATE = config.get('model.learning_rate')
        MAX_LENGTH = min(config.get('data.max_length'), opt_settings['max_length'])
        SAVE_DIR = config.get('model.output_dir')
        FEATURE_SELECTION = config.get('data.feature_selection')
        FEATURE_COUNT = config.get('data.feature_count')
        SAMPLE_FRAC = config.get('data.sample_fraction')
        MIXED_PRECISION = config.get('model.mixed_precision') or opt_settings['mixed_precision']
        CACHE_TOKENIZATION = config.get('data.cache_tokenization')
        EARLY_STOPPING = config.get('model.early_stopping')
        EVAL_STEPS = config.get('model.eval_steps')

        logger.info("Training Configuration:")
        logger.info(f"- Device: {hw_info['device']}")
        logger.info(f"- Batch size: {TRAIN_BATCH_SIZE}")
        logger.info(f"- Epochs: {EPOCHS}")
        logger.info(f"- Learning rate: {LEARNING_RATE}")
        logger.info(f"- Max sequence length: {MAX_LENGTH}")
        logger.info(f"- Sample fraction: {SAMPLE_FRAC}")
        logger.info(f"- Feature selection: {FEATURE_SELECTION} (features: {FEATURE_COUNT})")
        logger.info(f"- Mixed precision: {MIXED_PRECISION}")
        logger.info(f"- Cache tokenization: {CACHE_TOKENIZATION}")
        logger.info(f"- Early stopping patience: {EARLY_STOPPING}")
        logger.info(f"- Evaluation steps: {EVAL_STEPS}")
        
        # Save actual configuration used
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR, exist_ok=True)
        config_path = os.path.join(SAVE_DIR, 'training_config.json')
        config.save_config(config_path)
        logger.info(f"Saved configuration to {config_path}")
        
        # Track execution time
        start_time = time.time()
        
        # Load and prepare data with optimized feature selection
        logger.info(f"Loading data from {DATA_PATH}")
        data_loader = CyberDataLoader()
        try:
            texts, labels = data_loader.load_data(
                data_path=DATA_PATH,
                feature_selection=FEATURE_SELECTION,
                n_features=FEATURE_COUNT,
                sample_fraction=SAMPLE_FRAC
            )
            num_labels = data_loader.get_num_labels()
            logger.info(f"Loaded {len(texts)} samples with {num_labels} unique labels")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        # Create tokenizer
        logger.info(f"Loading tokenizer from {MODEL_PATH}")
        try:
            tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        # Create dataset with improved tokenization caching
        logger.info("Creating dataset")
        try:
            full_dataset = CyberSecurityDataset(
                texts, 
                labels, 
                tokenizer, 
                MAX_LENGTH, 
                cache_tokenization=CACHE_TOKENIZATION
            )
            
            # Get class weights directly from the dataset
            weight_tensor = full_dataset.get_class_weights().to(hw_info['device'])
            
            # Get label mappings for the model
            label_mapping = full_dataset.get_label_map()
            
            logger.info(f"Dataset created with {len(full_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        # Create model with fresh classification head and proper label mapping
        logger.info("Creating model")
        try:
            model = create_model_with_fresh_head(MODEL_PATH, num_labels, hw_info, data_loader, logger)
            
            # Store class weights in the model config as a list (JSON serializable)
            model.config.class_weights = weight_tensor.cpu().tolist()
            
            # Verify the classifier layer shape
            logger.info("Model classifier layer shape:")
            logger.info(f"Weight shape: {model.classifier.weight.shape}")
            logger.info(f"Bias shape: {model.classifier.bias.shape}")
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        # Split dataset with proper stratification for imbalanced data
        logger.info("Splitting dataset into train/validation sets")
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
        
        logger.info(f"Training on {train_size} samples, validating on {val_size} samples")
        
        # Create dataloaders
        logger.info("Creating data loaders")
        train_dataloader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        val_dataloader = DataLoader(val_dataset, **dataloader_kwargs)
        
        # Setup optimizer with weight decay
        logger.info("Setting up optimizer and scheduler")
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
        logger.info("Creating trainer")
        trainer = CyberBERTTrainer(model, tokenizer, hw_info['device'], logger)
        
        # Train with enhanced monitoring
        logger.info("Starting training")
        try:
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
            
            # Save training results
            results_file = os.path.join(SAVE_DIR, 'training_results.json')
            with open(results_file, 'w') as f:
                # Convert tensors to lists for JSON serialization
                serializable_results = {}
                for k, v in training_results.items():
                    if isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
                        serializable_results[k] = [float(item) for item in v]
                    else:
                        serializable_results[k] = v
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved training results to {results_file}")
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        # Display final execution time
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        # Save important features if feature selection was used
        if FEATURE_SELECTION and data_loader.selected_features:
            features_file = os.path.join(SAVE_DIR, 'selected_features.txt')
            with open(features_file, 'w') as f:
                for feature in data_loader.selected_features:
                    f.write(f"{feature}\n")
            logger.info(f"Saved {len(data_loader.selected_features)} selected features to {features_file}")
        
        # Save label mapping
        label_map_file = os.path.join(SAVE_DIR, 'label_mapping.json')
        with open(label_map_file, 'w') as f:
            json.dump({
                'id2label': {str(idx): label for idx, label in model.config.id2label.items()},
                'label2id': model.config.label2id
            }, f, indent=2)
        logger.info(f"Saved label mapping to {label_map_file}")
        
        logger.info("Training completed successfully")
        return 0

    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Fatal error: {str(e)}")
            logger.debug(traceback.format_exc())
        else:
            print(f"Fatal error before logger initialization: {str(e)}")
            print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
