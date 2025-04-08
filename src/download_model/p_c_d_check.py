from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("./models/cyberbert_model")
model = BertForSequenceClassification.from_pretrained("./models/cyberbert_model")

print("Model loaded successfully!")
