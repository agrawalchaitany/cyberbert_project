# Description: This script downloads the pre-trained BERT model and saves it locally.

from transformers import BertTokenizer, BertForSequenceClassification

# Define model name (pre-trained BERT)
MODEL_NAME = "distilbert-base-uncased"

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)  # Binary classification

# Save model locally
model.save_pretrained("./models/cyberbert_model")
tokenizer.save_pretrained("./models/cyberbert_model")

print("Model downloaded and saved in './models/cyberbert_model/'")
