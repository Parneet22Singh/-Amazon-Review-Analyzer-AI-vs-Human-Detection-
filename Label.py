import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load your dataset
df = pd.read_csv("dataset/reviews.csv")


# Clean up review text
df["Review"] = df["Review"].fillna("").astype(str)
df["Review"] = df["Review"].apply(lambda x: x.replace("\n", " ").strip())

# Load sentiment model (1–5 stars)
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=AutoModelForSequenceClassification.from_pretrained(sentiment_model_name),
    tokenizer=AutoTokenizer.from_pretrained(sentiment_model_name)
)

# Load AI detection model
ai_model_name = "roberta-large-mnli"
ai_pipeline = pipeline(
    "zero-shot-classification",
    model=AutoModelForSequenceClassification.from_pretrained(ai_model_name),
    tokenizer=AutoTokenizer.from_pretrained(ai_model_name)
)

# Sentiment scoring: 1–5 stars
def get_star_rating(text):
    if isinstance(text, str) and text.strip():
        result = sentiment_pipeline(text[:512])[0]
        return int(result['label'][0])  # From '5 stars' -> 5
    return None

# AI-likeness detection using zero-shot classification
def detect_ai_generated(text):
    if isinstance(text, str) and text.strip():
        result = ai_pipeline(
            text[:512],
            candidate_labels=["human", "AI"]
        )
        labels = result['labels']
        scores = result['scores']
        # Get confidence for 'AI' label
        ai_score = scores[labels.index("AI")]
        return ("AI" if ai_score > 0.5 else "human", round(ai_score, 3))
    return ("unknown", 0.0)

# Apply functions
df["rating"] = df["Review"].apply(get_star_rating)
df[["label", "score"]] = df["Review"].apply(detect_ai_generated).apply(pd.Series)

# Save results
df.to_csv("dataset/labelled_reviews_with_ai_detection.csv", index=False)
print("✅ Labeled reviews saved to dataset/labelled_reviews_with_ai_detection.csv")
