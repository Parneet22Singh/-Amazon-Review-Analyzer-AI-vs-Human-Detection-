import streamlit as st
import pandas as pd
import joblib
from Scraper import scrape_amazon_reviews
from collections import Counter
from textblob import TextBlob

# === Load model and vectorizer ===
model = joblib.load("dataset/trained.pkl")
vectorizer = joblib.load("dataset/vectorizer.pkl")

# === Title & Input ===
st.title("🛍️ Amazon Review Analyzer")
product_url = st.text_input("Enter Amazon Product Review URL:")

if product_url:
    try:
        df = scrape_amazon_reviews(product_url, max_pages=10)

        if df.empty:
            st.error("⚠️ No reviews found.")
        else:
            st.success(f"Scraped {len(df)} reviews.")

            # === Preprocess & Predict ===
            X = vectorizer.transform(df["Review"])
            predictions = model.predict(X)

            # Use predictions directly, capitalize for display
            df["Label"] = [str(p).capitalize() for p in predictions]  # "human" → "Human"

            # === Sentiment Analysis ===
            df["Sentiment"] = df["Review"].apply(lambda x: TextBlob(x).sentiment.polarity)

            # === Summary Metrics ===
            counts = Counter(df["Label"])
            total = len(df)
            ai_count = counts.get("Ai", 0)
            human_count = counts.get("Human", 0)
            ai_percent = round((ai_count / total) * 100, 2)
            human_percent = round((human_count / total) * 100, 2)
            avg_sentiment = df["Sentiment"].mean()

            st.subheader("🔍 Summary")
            st.write(f"🤖 AI-Generated Reviews: {ai_count} ({ai_percent}%)")
            st.write(f"🧠 Human-Written Reviews: {human_count} ({human_percent}%)")
            st.write(f"🌟 Average Sentiment Score: {avg_sentiment:.2f}")

            # === Suggestion Logic ===
            if ai_percent > 50:
                trust_note = "⚠️ Many reviews seem AI-generated. Take caution."
            elif avg_sentiment > 0.3:
                trust_note = "✅ Reviews are mostly positive. Likely a good buy."
            elif avg_sentiment < 0:
                trust_note = "❌ Many reviews are negative. Be cautious."
            else:
                trust_note = "🤔 Mixed reviews. Consider checking alternatives."

            st.subheader("📝 Purchase Suggestion")
            st.info(trust_note)

            # === Show Sample Reviews ===
            st.subheader("🗒️ Sample Reviews")
            st.write(df[["Review", "Label", "Sentiment"]].head(10))

    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
