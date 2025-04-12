import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import numpy as np
import logging
import json
from typing import Dict, Any, Optional, List
from torch.cuda.amp import autocast, GradScaler
import gc
import psutil

class CyberBERTTrainer:
    """
    Simplified trainer for CyberBERT model with core functionality intact.
    """
    
    def __init__(self, model, tokenizer, device, logger=None):
        """
        Initialize the trainer
        
        Args:
            model: The BERT model for sequence classification
            tokenizer: The tokenizer for the model
            device: The device to train on (cuda or cpu)
            logger: Optional logging.Logger instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': [],
        }
        
        # Set up logger
        self.logger = logger or logging.getLogger(__name__)
        
    def train(self, train_dataloader, val_dataloader, epochs, optimizer, scheduler, save_dir, 
              mixed_precision=True, early_stopping_patience=3, eval_steps=100):
        """
        Train the CyberBERT model
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            save_dir: Directory to save models
            mixed_precision: Use mixed precision training (faster on compatible GPUs)
            early_stopping_patience: Stop if no improvement for this many epochs
            eval_steps: Evaluate every this many steps (0 to disable)
        
        Returns:
            Dictionary with training history
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
            
        best_val_loss = float('inf')
        patience_counter = 0
        global_step = 0
        
        # Set up mixed precision training if available
        scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        self.logger.info(f"Training on device: {self.device}")
        self.logger.info(f"Training with mixed precision: {mixed_precision and self.device.type == 'cuda'}")
        
        # Force garbage collection before training
        gc.collect()
        
        try:
            for epoch in range(epochs):
                # Training
                self.model.train()
                total_train_loss = 0
                total_train_correct = 0
                total_train_samples = 0
                epoch_start_time = time.time()
                all_train_preds = []
                all_train_labels = []
                
                train_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
                
                for step, batch in enumerate(train_progress):
                    # Move tensors to device
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Handle both 'labels' and 'label' keys for backward compatibility
                    if 'labels' in batch:
                        labels = batch['labels'].to(self.device)
                    elif 'label' in batch:
                        labels = batch['label'].to(self.device)
                    else:
                        raise ValueError("Neither 'labels' nor 'label' found in batch data")
                    
                    # Calculate loss with mixed precision if enabled
                    if mixed_precision and self.device.type == 'cuda':
                        with autocast():
                            outputs = self.model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels
                            )
                            loss = outputs.loss
                            
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                    else:
                        # Standard forward/backward without mixed precision
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                    
                    # Track loss and accuracy
                    total_train_loss += loss.item()
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    correct = (predictions == labels).sum().item()
                    total_train_correct += correct
                    total_train_samples += labels.size(0)
                    
                    # Collect predictions and labels for f1 score
                    all_train_preds.extend(predictions.cpu().numpy())
                    all_train_labels.extend(labels.cpu().numpy())
                    
                    # Free memory after each training step
                    del input_ids, attention_mask, labels, outputs, logits, predictions
                    
                    # Update progress bar
                    train_acc = total_train_correct / total_train_samples
                    train_progress.set_postfix({
                        'loss': loss.item(), 
                        'acc': f'{train_acc:.4f}'
                    })
                    
                    # Increment global step
                    global_step += 1
                        
                    # Periodic validation during epoch
                    if eval_steps > 0 and global_step % eval_steps == 0:
                        # Check available memory before validation
                        available_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                        self.logger.info(f"Available memory before validation: {available_mem:.2f} GB")
                        
                        # Force garbage collection
                        gc.collect()
                        
                        # Quick validation with minimal batch count for memory-constrained systems
                        max_val_batches = None
                        if available_mem < 2.0:  # Less than 2GB free
                            max_val_batches = 10  # Limit validation to 10 batches
                            self.logger.warning(f"Limited memory available ({available_mem:.2f}GB), limiting validation to {max_val_batches} batches")
                            
                        val_loss, val_acc, val_f1 = self._quick_evaluate(val_dataloader, max_batches=max_val_batches)
                        self.logger.info(f"Step {global_step}: "
                                f"Train loss: {loss.item():.4f}, acc: {train_acc:.4f} | "
                                f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                        # Return to training mode
                        self.model.train()
                        
                        # Force garbage collection after validation
                        gc.collect()
                
                # Calculate epoch metrics
                avg_train_loss = total_train_loss / len(train_dataloader)
                train_acc = total_train_correct / total_train_samples
                
                # Calculate F1 score
                train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
                
                # Calculate epoch training time
                epoch_time = time.time() - epoch_start_time
                
                # Check if we have enough memory for full validation
                available_mem = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                self.logger.info(f"Available memory before full validation: {available_mem:.2f} GB")
                
                # Force garbage collection
                gc.collect()
                
                # Full validation
                val_metrics = self.evaluate(val_dataloader)
                val_loss = val_metrics['loss']
                val_acc = val_metrics['accuracy']
                val_f1 = val_metrics.get('f1_score', 0)
                
                # Add to history
                self.history['train_loss'].append(avg_train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_acc'].append(train_acc)
                self.history['val_acc'].append(val_acc)
                self.history['train_f1'].append(train_f1)
                self.history['val_f1'].append(val_f1)
                
                # Print epoch summary
                self.logger.info(f'Epoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s')
                self.logger.info(f'Train loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}, F1: {train_f1:.4f}')
                self.logger.info(f'Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}, F1: {val_f1:.4f}')
                
                # Save epoch history
                torch.save(self.history, os.path.join(save_dir, 'training_history.pt'))
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_path = os.path.join(save_dir, 'best_model')
                    os.makedirs(best_model_path, exist_ok=True)
                    
                    # Save model and configuration
                    self.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    
                    # Save metadata
                    metadata = {
                        'epoch': epoch + 1,
                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_acc),
                        'val_f1': float(val_f1),
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    with open(os.path.join(best_model_path, 'metadata.json'), 'w') as f:
                        json.dump(metadata, f, indent=2)
                        
                    self.logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self.logger.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                
                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
                
                # Force garbage collection between epochs
                gc.collect()
            
            # Save training plots
            self._save_training_plots(save_dir)
            
            return self.history
            
        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user")
            # Save checkpoint on interruption
            interrupted_path = os.path.join(save_dir, "interrupted_checkpoint")
            os.makedirs(interrupted_path, exist_ok=True)
            self.model.save_pretrained(interrupted_path)
            self.tokenizer.save_pretrained(interrupted_path)
            return self.history
        except Exception as e:
            self.logger.error(f"Training error: {str(e)}")
            # Try to save checkpoint on unexpected error
            try:
                error_path = os.path.join(save_dir, "error_checkpoint")
                os.makedirs(error_path, exist_ok=True)
                self.model.save_pretrained(error_path)
                self.tokenizer.save_pretrained(error_path)
                self.logger.info(f"Saved checkpoint at error to {error_path}")
            except:
                self.logger.error("Failed to save error checkpoint")
            raise
    
    def _quick_evaluate(self, dataloader, max_batches=None):
        """
        Quick evaluation for periodic validation during training - memory-optimized version
        
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        # Memory optimization - explicitly collect garbage before evaluation
        gc.collect()
        
        # Check if we're on a memory-constrained system
        is_memory_constrained = psutil.virtual_memory().available < 2 * 1024 * 1024 * 1024  # < 2GB free
        
        # For very memory-constrained environments, process one sample at a time
        effective_batch_size = 1 if is_memory_constrained else None
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                
                # For memory-constrained systems, process the batch one sample at a time
                if effective_batch_size == 1 and len(batch['input_ids']) > 1:
                    # Process one sample at a time
                    for j in range(len(batch['input_ids'])):
                        input_ids = batch['input_ids'][j:j+1].to(self.device)
                        attention_mask = batch['attention_mask'][j:j+1].to(self.device)
                        
                        # Handle both 'labels' and 'label' keys
                        if 'labels' in batch:
                            labels = batch['labels'][j:j+1].to(self.device)
                        elif 'label' in batch:
                            labels = batch['label'][j:j+1].to(self.device)
                        else:
                            raise ValueError("Neither 'labels' nor 'label' found in batch data")
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        loss = outputs.loss
                        total_val_loss += loss.item()
                        
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        correct += (predictions == labels).sum().item()
                        total += labels.size(0)
                        
                        # Collect for F1 score
                        all_preds.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        # Memory optimization - clear cache after each sample
                        del input_ids, attention_mask, labels, outputs, logits, predictions
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # Process the entire batch at once
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Handle both 'labels' and 'label' keys
                    if 'labels' in batch:
                        labels = batch['labels'].to(self.device)
                    elif 'label' in batch:
                        labels = batch['label'].to(self.device)
                    else:
                        raise ValueError("Neither 'labels' nor 'label' found in batch data")
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_val_loss += loss.item()
                    
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    correct += (predictions == labels).sum().item()
                    total += labels.size(0)
                    
                    # Collect for F1 score
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Memory optimization - clear cache after each batch
                    del input_ids, attention_mask, labels, outputs, logits, predictions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Periodically force garbage collection
                if i % 5 == 0:  # Every 5 batches
                    gc.collect()
        
        # Calculate metrics
        loss = total_val_loss / min(len(dataloader), max_batches or float('inf'))
        acc = correct / total if total > 0 else 0
        
        # Handle case where we might not have enough data for F1 calculation
        if len(all_preds) > 0 and len(np.unique(all_labels)) > 1:
            f1 = f1_score(all_labels, all_preds, average='weighted')
        else:
            f1 = 0
        
        return loss, acc, f1
                
    def evaluate(self, dataloader):
        """
        Full evaluation with detailed metrics, memory-optimized for resource-constrained systems
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_val_loss = 0
        all_predictions = []
        all_labels = []
        
        # Memory optimization - collect garbage before evaluation
        gc.collect()
        
        # Use smaller evaluation batch sizes for memory-constrained environments
        is_memory_constrained = psutil.virtual_memory().available < 2 * 1024 * 1024 * 1024  # < 2GB free
        
        # For very memory-constrained systems, limit number of batches
        max_batches = None
        if is_memory_constrained:
            available_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
            if available_gb < 1.0:  # Less than 1GB available
                max_batches = 20
                self.logger.warning(f"Very limited memory ({available_gb:.2f}GB), limiting evaluation to {max_batches} batches")
            else:
                max_batches = 50
                self.logger.warning(f"Limited memory ({available_gb:.2f}GB), limiting evaluation to {max_batches} batches")
                
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Check if we've reached max batches
                if max_batches is not None and batch_count >= max_batches:
                    break
                    
                # For memory-constrained systems, process the batch one sample at a time
                if is_memory_constrained and len(batch['input_ids']) > 1:
                    # Process one sample at a time
                    for j in range(len(batch['input_ids'])):
                        input_ids = batch['input_ids'][j:j+1].to(self.device)
                        attention_mask = batch['attention_mask'][j:j+1].to(self.device)
                        
                        # Handle both 'labels' and 'label' keys
                        if 'labels' in batch:
                            labels = batch['labels'][j:j+1].to(self.device)
                        elif 'label' in batch:
                            labels = batch['label'][j:j+1].to(self.device)
                        else:
                            raise ValueError("Neither 'labels' nor 'label' found in batch data")
                        
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        
                        total_val_loss += outputs.loss.item()
                        
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=-1)
                        
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                        
                        # Memory optimization - clear cache after each sample
                        del input_ids, attention_mask, labels, outputs, logits, predictions
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    # Process the entire batch
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    # Handle both 'labels' and 'label' keys for backward compatibility
                    if 'labels' in batch:
                        labels = batch['labels'].to(self.device)
                    elif 'label' in batch:
                        labels = batch['label'].to(self.device)
                    else:
                        raise ValueError("Neither 'labels' nor 'label' found in batch data")
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    total_val_loss += outputs.loss.item()
                    
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    # Memory optimization - clear cache after each batch
                    del input_ids, attention_mask, labels, outputs, logits, predictions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Increment batch counter
                batch_count += 1
                
                # Force garbage collection periodically
                if batch_count % 5 == 0:  # Every 5 batches
                    gc.collect()
        
        # Handle case where we limited the number of batches
        divisor = min(batch_count, len(dataloader))
        
        # Calculate metrics
        val_loss = total_val_loss / divisor if divisor > 0 else 0
        
        # Make sure we have enough valid predictions
        if len(all_predictions) == 0 or len(all_labels) == 0:
            self.logger.warning("No valid predictions or labels obtained during evaluation")
            return {'loss': val_loss, 'accuracy': 0, 'f1_score': 0}
            
        # Calculate accuracy and F1 score
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Check for edge cases that could cause F1 calculation to fail
        if len(np.unique(all_labels)) <= 1:
            self.logger.warning("Not enough unique classes for F1 score calculation")
            f1 = 0
        else:
            f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'loss': val_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        # Classification report - only if we have enough data
        if len(np.unique(all_labels)) > 1 and len(all_labels) >= 10:
            try:
                class_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
                report = classification_report(all_labels, all_predictions, target_names=class_names)
                self.logger.info("\nClassification Report:\n" + report)
            except Exception as e:
                self.logger.warning(f"Could not generate classification report: {str(e)}")
        
        return metrics
    
    def _save_training_plots(self, save_dir):
        """Generate and save training progress plots"""
        try:
            os.makedirs(save_dir, exist_ok=True)
                
            # Create figure
            plt.figure(figsize=(12, 10))
            
            # Plot training and validation loss
            plt.subplot(2, 2, 1)
            plt.plot(self.history['train_loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Training and Validation Loss')
            
            # Plot training and validation accuracy
            plt.subplot(2, 2, 2)
            plt.plot(self.history['train_acc'], label='Training Accuracy')
            plt.plot(self.history['val_acc'], label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title('Training and Validation Accuracy')
            
            # Plot training and validation F1 score
            plt.subplot(2, 2, 3)
            plt.plot(self.history['train_f1'], label='Training F1')
            plt.plot(self.history['val_f1'], label='Validation F1')
            plt.xlabel('Epochs')
            plt.ylabel('F1 Score')
            plt.legend()
            plt.title('Training and Validation F1 Score')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300)
            plt.close()
                
        except Exception as e:
            self.logger.error(f"Error saving training plots: {str(e)}")
        
    def predict(self, texts, max_length=None, batch_size=8):
        """
        Make predictions on new texts
        
        Args:
            texts: List of texts to predict
            max_length: Maximum sequence length (None uses model's default)
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary with predictions and probabilities
        """
        if max_length is None:
            max_length = 256  # reasonable default
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the input texts
            encodings = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Get probabilities and predicted classes
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy()
            
            # Convert to class names
            predicted_labels = [self.model.config.id2label[class_id] for class_id in predicted_class_ids]
            
            all_predictions.extend(predicted_labels)
            all_probabilities.extend(probs.cpu().numpy())
        
        return {
            'predictions': all_predictions,
            'probabilities': all_probabilities
        }
