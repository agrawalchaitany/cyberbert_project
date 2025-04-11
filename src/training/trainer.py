import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

class CyberBERTTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def train(self, train_dataloader, val_dataloader, epochs, optimizer, scheduler, save_dir, 
              mixed_precision=True, early_stopping_patience=3, eval_steps=100):
        """
        Enhanced training with mixed precision, early stopping, and detailed metrics
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Number of training epochs
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            save_dir: Directory to save models
            mixed_precision: Use mixed precision training (faster on compatible GPUs)
            early_stopping_patience: Stop if no improvement for this many epochs
            eval_steps: Evaluate on validation set every this many steps
        """
        best_val_loss = float('inf')
        patience_counter = 0
        total_train_time = 0
        scaler = GradScaler() if mixed_precision and self.device.type == 'cuda' else None
        
        print(f"\nTraining on device: {self.device}")
        print(f"Training with mixed precision: {mixed_precision and self.device.type == 'cuda'}")
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            total_train_correct = 0
            total_train_samples = 0
            epoch_start_time = time.time()
            
            train_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for step, batch in enumerate(train_progress):
                # Clear previous gradients
                optimizer.zero_grad()
                
                # Move tensors to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass with mixed precision for compatible GPUs
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
                
                # Step the scheduler
                scheduler.step()
                
                # Track loss and accuracy
                total_train_loss += loss.item()
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct = (predictions == labels).sum().item()
                total_train_correct += correct
                total_train_samples += labels.size(0)
                
                # Update progress bar
                train_acc = total_train_correct / total_train_samples
                train_progress.set_postfix({
                    'loss': loss.item(), 
                    'acc': f'{train_acc:.4f}',
                    'lr': scheduler.get_last_lr()[0]
                })
                
                # Periodic validation during epoch
                if eval_steps > 0 and step > 0 and step % eval_steps == 0:
                    # Quick validation
                    val_loss, val_acc = self._quick_evaluate(val_dataloader)
                    print(f"\nStep {step}/{len(train_dataloader)}: "
                          f"Train loss: {loss.item():.4f}, acc: {train_acc:.4f} | "
                          f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")
                    # Return to training mode
                    self.model.train()
            
            # Calculate epoch metrics
            avg_train_loss = total_train_loss / len(train_dataloader)
            train_acc = total_train_correct / total_train_samples
            
            # Calculate epoch training time
            epoch_time = time.time() - epoch_start_time
            total_train_time += epoch_time
            
            # Full validation
            val_metrics = self.evaluate(val_dataloader, compute_confusion=True)
            val_loss = val_metrics['loss']
            val_acc = val_metrics['accuracy']
            
            # Add to history
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{epochs} - Time: {epoch_time:.2f}s')
            print(f'Average training loss: {avg_train_loss:.4f}, accuracy: {train_acc:.4f}')
            print(f'Validation loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                # Save model and configuration
                self.model.save_pretrained(os.path.join(save_dir, 'best_model'))
                self.tokenizer.save_pretrained(os.path.join(save_dir, 'best_model'))
                
                # Also save training history and metrics
                history_path = os.path.join(save_dir, 'training_history.pt')
                torch.save(self.history, history_path)
                
                print(f"Saved best model with validation loss: {best_val_loss:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
            
            # Check for early stopping
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Print final training summary
        print(f"\nTraining completed in {total_train_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save training plots
        self._save_training_plots(save_dir)
        
        return self.history
    
    def _quick_evaluate(self, dataloader, max_batches=None):
        """Quick evaluation for periodic validation during training"""
        self.model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
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
        
        return total_val_loss / min(len(dataloader), max_batches or float('inf')), correct / total
                
    def evaluate(self, dataloader, compute_confusion=False):
        """Full evaluation with detailed metrics"""
        self.model.eval()
        total_val_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
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
        
        metrics = {
            'loss': val_loss,
            'accuracy': accuracy,
        }
        
        # Add detailed classification metrics if requested
        if compute_confusion:
            class_report = classification_report(
                all_labels, 
                all_predictions, 
                target_names=[self.model.config.id2label[i] for i in range(len(self.model.config.id2label))],
                output_dict=True
            )
            metrics['classification_report'] = class_report
            
            cm = confusion_matrix(all_labels, all_predictions)
            metrics['confusion_matrix'] = cm
            
            # Print classification report
            print("\nClassification Report:")
            print(classification_report(
                all_labels, 
                all_predictions, 
                target_names=[self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
            ))
        
        return metrics
    
    def _save_training_plots(self, save_dir):
        """Generate and save training progress plots"""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        plt.figure(figsize=(12, 4))
        
        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'))
        plt.close()
        
    def predict(self, texts, max_length=None):
        """Make predictions on new texts"""
        if max_length is None:
            max_length = self.model.config.max_position_embeddings
            
        # Tokenize the input texts
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
        # Get predicted class indices
        logits = outputs.logits
        predicted_class_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        
        # Convert to class names
        predicted_labels = [self.model.config.id2label[class_id] for class_id in predicted_class_ids]
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()
        
        return {
            'predictions': predicted_labels,
            'probabilities': probabilities,
            'class_ids': predicted_class_ids
        }
