from flask import Flask, request, jsonify
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Load the sentiment model
print("[INFO] Loading sentiment model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
bert_model.load_state_dict(torch.load(r"bert_sentiment_model.pth", map_location=device))
bert_model.to(device)
bert_model.eval()

# Load the rating model
print("[INFO] Loading rating model...")
checkpoint = torch.load(r"rf_rating_model.pth")
vectorizer = checkpoint["vectorizer_state"]
rf_model = checkpoint["model_state"]

# Load tokenizer
print("[INFO] Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# Helper function for sentiment prediction
def predict_sentiment(text):
    encoding = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_mask)
        sentiment_pred = torch.argmax(outputs.logits, dim=1).item()

    sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_map[sentiment_pred]


# Helper function for rating prediction
def predict_rating(text):
    features = vectorizer.transform([text])
    predicted_rating = rf_model.predict(features)[0]
    return round(predicted_rating, 2)


@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    # Check if `review_text` is provided
    if "review_text" not in data:
        return jsonify({"error": "Missing 'review_text' field in the request"}), 400

    review_text = data["review_text"]

    # Predict sentiment and rating
    sentiment = predict_sentiment(review_text)
    rating = predict_rating(review_text)

    response = {
        "review_text": review_text,
        "predicted_sentiment": sentiment,
        "predicted_rating": rating
    }

    return jsonify(response), 200


if __name__ == '__main__':
    app.run(debug=True)
