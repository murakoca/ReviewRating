import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Load and preprocess the dataset
print("[INFO] Loading dataset...")
csv_file = r"customer_reviews.csv"
df = pd.read_csv(csv_file)

print("[INFO] Preprocessing dataset...")
df = df.dropna(subset=["review_text", "rating"])
df["sentiment"] = df["rating"].apply(lambda x: "positive" if x >= 5 else "negative" if x <= 2 else "neutral")

# Map sentiment to labels for classification
df["sentiment_label"] = df["sentiment"].map({"positive": 2, "neutral": 1, "negative": 0})

# Step 2: Split into train/test sets
print("[INFO] Splitting dataset...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["sentiment_label"])

# Separate features and targets
train_texts, train_sentiments, train_ratings = train_df["review_text"], train_df["sentiment_label"], train_df["rating"]
val_texts, val_sentiments, val_ratings = val_df["review_text"], val_df["sentiment_label"], val_df["rating"]

# Step 3: Initialize BERT tokenizer
print("[INFO] Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define Dataset class for sentiment classification
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        label = self.labels.iloc[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

# Create DataLoader
def create_data_loader(texts, labels, tokenizer, batch_size=16):
    dataset = SentimentDataset(texts, labels, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

batch_size = 16
train_loader = create_data_loader(train_texts, train_sentiments, tokenizer, batch_size)
val_loader = create_data_loader(val_texts, val_sentiments, tokenizer, batch_size)

# Step 4: Initialize BERT model for sentiment classification
print("[INFO] Initializing BERT model...")
bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3  # Positive, Neutral, Negative
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)

# Compute class weights
class_weights = compute_class_weight("balanced", classes=np.unique(train_sentiments), y=train_sentiments)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Optimizer
optimizer = AdamW(bert_model.parameters(), lr=2e-5, eps=1e-8)

# Training function for BERT
def train_bert_epoch(model, data_loader, optimizer, device, class_weights):
    model.train()
    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training BERT"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

        loss = outputs.loss
        loss = loss * class_weights[labels].mean()
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        correct_predictions += (predictions == labels).sum().item()
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    return correct_predictions / len(data_loader.dataset), np.mean(losses)

# Evaluation function for BERT
def evaluate_bert(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating BERT"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            loss = outputs.loss
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            correct_predictions += (predictions == labels).sum().item()
            losses.append(loss.item())

    return correct_predictions / len(data_loader.dataset), np.mean(losses)

# Train BERT for sentiment classification
print("[INFO] Training sentiment model with BERT...")
epochs = 2
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    train_acc, train_loss = train_bert_epoch(bert_model, train_loader, optimizer, device, class_weights)
    val_acc, val_loss = evaluate_bert(bert_model, val_loader, device)

    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

torch.save(bert_model.state_dict(), "bert_sentiment_model.pth")
print("[INFO] Sentiment model saved.")

# Step 5: Train Random Forest for rating prediction
print("[INFO] Training Random Forest for rating prediction...")
vectorizer = TfidfVectorizer(max_features=5000)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

X_train_tfidf = vectorizer.fit_transform(train_texts)
X_val_tfidf = vectorizer.transform(val_texts)

rf_model.fit(X_train_tfidf, train_ratings)
print("[INFO] Rating model training completed.")

# Evaluate rating prediction model
rf_predictions = rf_model.predict(X_val_tfidf)
print("\n[INFO] Rating Prediction Evaluation:")
print(f"Mean Absolute Error: {mean_absolute_error(val_ratings, rf_predictions):.2f}")

# Save TF-IDF vectorizer and Random Forest model to .pth
torch.save({
    "vectorizer_state": vectorizer,
    "model_state": rf_model
}, "rf_rating_model.pth")

print("[INFO] Rating model and vectorizer saved.")
