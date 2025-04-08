import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

class CyberBERTTrainer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def train(self, train_dataloader, val_dataloader, epochs, optimizer, scheduler, save_dir):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            train_progress = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')
            
            for batch in train_progress:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_progress.set_postfix({'loss': loss.item()})
                
            avg_train_loss = total_train_loss / len(train_dataloader)
            
            # Validation
            val_loss = self.evaluate(val_dataloader)
            
            print(f'Epoch {epoch+1}:')
            print(f'Average training loss: {avg_train_loss:.4f}')
            print(f'Validation loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                self.model.save_pretrained(os.path.join(save_dir, 'best_model'))
                self.tokenizer.save_pretrained(os.path.join(save_dir, 'best_model'))
                
    def evaluate(self, dataloader):
        self.model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
                
        return total_val_loss / len(dataloader)
