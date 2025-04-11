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
                        # Quick validation
                        val_loss, val_acc, val_f1 = self._quick_evaluate(val_dataloader)
                        self.logger.info(f"Step {global_step}: "
                                f"Train loss: {loss.item():.4f}, acc: {train_acc:.4f} | "
                                f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}, F1: {val_f1:.4f}")
                        # Return to training mode
                        self.model.train()
                
                # Calculate epoch metrics
                avg_train_loss = total_train_loss / len(train_dataloader)
                train_acc = total_train_correct / total_train_samples
                
                # Calculate F1 score
                train_f1 = f1_score(all_train_labels, all_train_preds, average='weighted')
                
                # Calculate epoch training time
                epoch_time = time.time() - epoch_start_time
                
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
    
    def _quick_evaluate(self, dataloader, max_batches=None):
        """
        Quick evaluation for periodic validation during training
        
        Returns:
            Tuple of (loss, accuracy, f1_score)
        """
        self.model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                    
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
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                # Collect for F1 score
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        loss = total_val_loss / min(len(dataloader), max_batches or float('inf'))
        acc = correct / total if total > 0 else 0
        f1 = f1_score(all_labels, all_preds, average='weighted') if len(all_preds) > 0 else 0
        
        return loss, acc, f1
                
    def evaluate(self, dataloader):
        """
        Full evaluation with detailed metrics
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_val_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
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
        
        # Calculate metrics
        val_loss = total_val_loss / len(dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'loss': val_loss,
            'accuracy': accuracy,
            'f1_score': f1
        }
        
        # Classification report
        if len(np.unique(all_labels)) > 1:
            class_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            report = classification_report(all_labels, all_predictions, target_names=class_names)
            self.logger.info("\nClassification Report:\n" + report)
        
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
