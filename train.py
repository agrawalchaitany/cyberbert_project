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
from dotenv import load_dotenv

# Add project root to path for imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import project modules
from src.data.dataset import CyberSecurityDataset
from src.training.trainer import CyberBERTTrainer
from src.data.data_loader import CyberDataLoader
from src.utils.logger import Logger
from src.utils.config import Config
from src.utils.system_monitor import SystemMonitor, monitor_and_log_system

def check_cuda_available():
    """Check if CUDA is available on the system without requiring torch"""
    try:
        # Try importing CUDA toolkit
        import ctypes
        ctypes.CDLL("nvcuda.dll" if platform.system() == "Windows" else "libcuda.so.1")
        return True
    except:
        return False

def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information and determine optimal device for training"""
    import torch
    
    info = {
        'device': None,
        'batch_size': 8,
        'memory_gb': 0,
        'description': '',
        'supports_cuda': False,
        'supports_mps': False,  # For Apple Silicon
        'total_ram': psutil.virtual_memory().total / 1e9,  # GB
        'cpu_count': psutil.cpu_count(logical=False) or 1,
        'cpu_threads': psutil.cpu_count(logical=True) or 2
    }
    
    # Check for CUDA first
    if torch.cuda.is_available():
        info['device'] = torch.device('cuda')
        info['supports_cuda'] = True
        info['memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info['gpu_name'] = torch.cuda.get_device_name(0)
        info['batch_size'] = min(32, max(4, int(info['memory_gb'] / 1.5)))
        info['description'] = f"GPU: {info['gpu_name']} ({info['memory_gb']:.1f}GB)"
    # Then check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = torch.device('mps')
        info['supports_mps'] = True
        info['memory_gb'] = info['total_ram']  # Use system RAM as estimate
        info['batch_size'] = min(16, max(2, int(info['total_ram'] / 3)))
        info['description'] = f"Apple Silicon MPS ({info['memory_gb']:.1f}GB RAM)"
    # Default to CPU
    else:
        info['device'] = torch.device('cpu')
        info['memory_gb'] = info['total_ram']
        info['batch_size'] = min(8, max(1, int(info['total_ram'] / 4)))
        info['description'] = f"CPU: {platform.processor()}"
    
    # Set optimized parameters based on device
    info['half_precision'] = info['supports_cuda']  # Use half precision on CUDA
    info['gradient_checkpointing'] = info['memory_gb'] < 12  # Use gradient checkpointing on lower memory
    info['worker_threads'] = min(4, max(0, info['cpu_threads'] - 2))  # Leave some cores for system
    
    return info

def optimize_memory(hw_info: Dict[str, Any], model=None):
    """Apply memory optimizations based on hardware"""
    gc.collect()
    
    # CUDA-specific optimizations
    if hw_info['supports_cuda']:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        if model:
            model.cuda()
            torch.cuda.empty_cache()
    # MPS-specific optimizations (Apple Silicon)
    elif hw_info['supports_mps']:
        if model:
            model.to(hw_info['device'])
    # CPU-specific optimizations
    else:
        torch.set_num_threads(hw_info['cpu_threads'])
        if model:
            model.cpu()
    
    # Enable gradient checkpointing if available and needed
    if model and hasattr(model, 'gradient_checkpointing_enable') and hw_info['gradient_checkpointing']:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

def print_hardware_summary(hw_info: Dict[str, Any]):
    """Print hardware configuration summary"""
    print("\nHardware Configuration:")
    print(f"- {hw_info['description']}")
    print(f"- CPU Cores: {hw_info['cpu_count']} (Threads: {hw_info['cpu_threads']})")
    print(f"- Available Memory: {hw_info['memory_gb']:.1f} GB")
    print(f"- Device Type: {hw_info['device'].type}")
    
    if hw_info['supports_cuda']:
        print(f"- CUDA Available: Yes")
    elif hw_info.get('supports_mps', False):
        print(f"- Apple Silicon MPS Available: Yes")
    
    print(f"- Recommended Batch Size: {hw_info['batch_size']}")
    print(f"- Gradient Checkpointing: {'Enabled' if hw_info['gradient_checkpointing'] else 'Disabled'}")
    print(f"- Half Precision: {'Enabled' if hw_info['half_precision'] else 'Disabled'}")
    print(f"- DataLoader Workers: {hw_info['worker_threads']}")

def get_optimized_settings(hw_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate optimal hyperparameters based on hardware configuration
    """
    settings = {
        # Training settings
        'batch_size': hw_info['batch_size'],
        'max_length': 128,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'mixed_precision': hw_info['supports_cuda'],
        'gradient_accumulation_steps': 1,
        
        # DataLoader settings
        'num_workers': hw_info['worker_threads'],
        'pin_memory': hw_info['supports_cuda'],
        
        # Trainer settings
        'eval_steps': 100 if hw_info['supports_cuda'] else 500,
        'warmup_ratio': 0.1,
        'gradient_checkpointing': hw_info['gradient_checkpointing'],
        'early_stopping_patience': 3,
    }
    
    # Adjust for different device types
    if hw_info['supports_cuda']:
        # On CUDA, we can use larger batches and faster evaluation
        settings['learning_rate'] = 5e-5  # Slightly higher learning rate
    elif hw_info.get('supports_mps', False):
        # Apple Silicon specifics
        settings['mixed_precision'] = False  # MPS doesn't support mixed precision yet
    else:
        # CPU optimizations
        settings['max_length'] = 96  # Even shorter sequences on CPU
        settings['learning_rate'] = 1e-5  # Lower learning rate for stability
        settings['batch_size'] = max(1, settings['batch_size'])  # Ensure batch size is at least 1
        
    return settings

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
    
    # System monitoring arguments
    parser.add_argument('--monitor-system', action='store_true', help='Enable system monitoring during training')
    parser.add_argument('--monitor-interval', type=float, default=30.0, help='System monitoring interval in seconds')
    
    return parser.parse_args()

def create_model_with_fresh_head(model_path: str, num_labels: int, hw_info: Dict[str, Any], 
                                data_loader: CyberDataLoader, logger, data_path: str = None) -> BertForSequenceClassification:
    """
    Create a model with a fresh classification head for cybersecurity data
    
    Args:
        model_path: Path to pre-trained BERT model
        num_labels: Number of classification labels
        hw_info: Hardware information
        data_loader: Data loader with label information
        logger: Logger instance
        data_path: Path to dataset (optional)
    
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
        try:
            # First try loading config from model path
            config = BertConfig.from_pretrained(model_path)
            # Update config with our classification settings
            config.num_labels = num_labels
            config.hidden_dropout_prob = 0.2
            config.classifier_dropout = 0.2  
            config.id2label = id2label
            config.label2id = label2id
        except Exception as config_error:
            logger.warning(f"Could not load config from {model_path}, creating new config: {str(config_error)}")
            # Create config from scratch if loading fails
            config = BertConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.2,
                attention_probs_dropout_prob=0.1,
                num_labels=num_labels,
                classifier_dropout=0.2,
                id2label=id2label,
                label2id=label2id
            )
        
        # Add training metadata to config
        config.task_specific_params = {
            "cybersecurity_classification": {
                "num_labels": num_labels,
                "created_with": "CyberBERT Trainer",
                "training_data": os.path.basename(data_path) if data_path else "unknown",
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
        # Load environment variables from .env file
        load_dotenv()
        
        # Parse arguments
        global args
        args = parse_args()
        
        # Initialize configuration
        config = Config(args.config if args.config else None)
        config.update_from_args(args)
        
        # Override with environment variables if available
        if args.data is None and "DATA_PATH" in os.environ:
            config.set('data.data_path', os.environ.get("DATA_PATH", "data/processed/clean_data.csv"))
            
        if args.model is None and "MODEL_NAME" in os.environ:
            config.set('model.model_path', f"models/cyberbert_model")
            
        if args.epochs is None:
            # Use environment variable EPOCHS or CPU_EPOCHS based on hardware
            if torch.cuda.is_available():
                config.set('model.epochs', int(os.environ.get("EPOCHS", 10)))
            else:
                config.set('model.epochs', int(os.environ.get("CPU_EPOCHS", 3)))
                
        if args.batch_size is None:
            # Use environment variable BATCH_SIZE or CPU_BATCH_SIZE based on hardware
            if torch.cuda.is_available():
                config.set('model.batch_size', int(os.environ.get("BATCH_SIZE", 32)))
            else:
                config.set('model.batch_size', int(os.environ.get("CPU_BATCH_SIZE", 8)))
                
        if args.max_length is None:
            # Use environment variable MAX_LENGTH or CPU_MAX_LENGTH based on hardware
            if torch.cuda.is_available():
                config.set('data.max_length', int(os.environ.get("MAX_LENGTH", 256)))
            else:
                config.set('data.max_length', int(os.environ.get("CPU_MAX_LENGTH", 128)))
                
        if args.feature_count is None:
            # Use environment variable FEATURE_COUNT or CPU_FEATURE_COUNT based on hardware
            if torch.cuda.is_available():
                config.set('data.feature_count', int(os.environ.get("FEATURE_COUNT", 40)))
            else:
                config.set('data.feature_count', int(os.environ.get("CPU_FEATURE_COUNT", 20)))
        
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
        
        # Initialize system monitoring if enabled
        system_monitor = None
        if args.monitor_system:
            logger.info(f"Starting system monitoring with interval of {args.monitor_interval} seconds")
            metrics_dir = os.path.join(config.get('model.output_dir', 'models/output'), 'metrics')
            system_monitor = monitor_and_log_system(logger, metrics_dir, args.monitor_interval)
            logger.info(f"System monitoring activated. Metrics will be saved to {metrics_dir}")
        
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
        # Ensure model path is valid - use a default if needed
        if MODEL_PATH is None:
            MODEL_PATH = 'models/cyberbert_model'
            logger.warning(f"Model path was not specified, using default: {MODEL_PATH}")
            
        DATA_PATH = config.get('data.data_path')
        
        # Set appropriate epochs based on device
        EPOCHS = config.get('model.epochs', 5)  # Default to 5 if not specified
        if hw_info['device'].type == 'cpu':  # Use .type property instead of comparing with string
            # If epochs not explicitly set but running on CPU, use a lower value
            if args.epochs is None:
                logger.warning(f"Running on CPU: Using CPU-optimized epochs setting from .env file")
        
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
        
        # Create dataset with tokenization caching
        logger.info("Creating dataset")
        try:
            # Set up cache directory if tokenization caching is enabled
            cache_dir = "cache" if CACHE_TOKENIZATION else None
            
            full_dataset = CyberSecurityDataset(
                texts, 
                labels, 
                tokenizer_name_or_path=MODEL_PATH,
                max_length=MAX_LENGTH,
                cache_dir=cache_dir
            )
            
            # Get class weights directly from the dataset
            weight_tensor = full_dataset.get_class_weights().to(hw_info['device'])
            
            # Get label mappings for the model
            label_mapping = full_dataset.get_label_mapping()
            
            logger.info(f"Dataset created with {len(full_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create dataset: {str(e)}")
            logger.debug(traceback.format_exc())
            raise
        
        # Create model with fresh classification head and proper label mapping
        logger.info("Creating model")
        try:
            model = create_model_with_fresh_head(MODEL_PATH, num_labels, hw_info, data_loader, logger, DATA_PATH)
            
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
        finally:
            # Stop system monitoring if it was enabled
            if system_monitor:
                logger.info("Stopping system monitoring")
                system_monitor.stop()
                logger.info("System monitoring stopped")
        
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
        # Stop system monitoring in case of error
        if 'system_monitor' in locals() and system_monitor:
            try:
                system_monitor.stop()
                if 'logger' in locals():
                    logger.info("System monitoring stopped due to error")
            except:
                pass
                
        if 'logger' in locals():
            logger.error(f"Fatal error: {str(e)}")
            logger.debug(traceback.format_exc())
        else:
            print(f"Fatal error before logger initialization: {str(e)}")
            print(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
