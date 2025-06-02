Certainly! Here’s a complete, updated README incorporating all parts including scraping, labeling, training, and the Streamlit app:

---

# Amazon Review Analyzer

A complete pipeline to scrape Amazon product reviews, automatically label them as human-written or AI-generated, analyze sentiment, and provide insights via an interactive Streamlit app.

---

## Features

* Scrape multiple pages of Amazon product reviews using Selenium (`Scraper.py`).
* Automatically label reviews as **AI-generated** or **Human-written** using zero-shot classification (`label.py`).
* Perform sentiment analysis to assign star ratings (1 to 5) to each review.
* Train a Logistic Regression model on labeled data to detect AI-generated reviews (`Train_model.py`).
* Interactive Streamlit app (`app.py`) that:

  * Takes an Amazon product review URL as input.
  * Scrapes reviews, predicts AI/human labels, and analyzes sentiment.
  * Displays summary metrics and purchase suggestions.
  * Shows sample reviews with labels and sentiment scores.

---

## Project Structure

```
dataset/
  ├── reviews.csv                         # Raw scraped reviews
  ├── labelled_reviews_with_ai_detection.csv  # Labeled dataset with sentiment and AI labels
  ├── trained.pkl                         # Trained Logistic Regression model
  ├── vectorizer.pkl                      # Text vectorizer used for model training
Scraper.py                               # Scrapes Amazon reviews
label.py                                # Labels scraped reviews with AI/human and sentiment
Train_model.py                          # Trains model on labeled reviews
app.py                                  # Streamlit app for review analysis
chromedriver.exe                        # Chrome driver for Selenium scraping
```

---

## Setup & Usage

### 1. Scrape Reviews

Run `Scraper.py` with the target Amazon product review URL to scrape up to 10 pages of reviews. Reviews are saved to:

```
dataset/reviews.csv
```

### 2. Label Reviews

Run `label.py` to automatically label each review as **AI-generated** or **Human-written**, and assign a star rating (1–5) based on sentiment. This uses pretrained Hugging Face models:

* `nlptown/bert-base-multilingual-uncased-sentiment` for sentiment.
* `roberta-large-mnli` for AI detection via zero-shot classification.

Output:

```
dataset/labelled_reviews_with_ai_detection.csv
```

### 3. Train Model

Run `Train_model.py` to train a Logistic Regression model on the labeled data. The model and vectorizer are saved as:

```
dataset/trained.pkl
dataset/vectorizer.pkl
```

### 4. Run Streamlit App

Run the Streamlit app with:

```bash
streamlit run app.py
```

Enter an Amazon product review URL. The app will:

* Scrape reviews,
* Predict AI or human labels,
* Analyze sentiment,
* Show summary statistics,
* Provide purchase recommendations,
* Display sample reviews.

---

## Key Components

### Scraper.py

* Uses Selenium with Chrome in headless mode.
* Maintains user profile to avoid repeated login prompts.
* Extracts review text from multiple pages.

### label.py

* Cleans review text.
* Uses pretrained transformers for sentiment scoring (1–5 stars).
* Uses zero-shot classification for AI detection.
* Outputs a labeled CSV file for training.

### Train\_model.py

* Loads labeled reviews.
* Cleans and vectorizes text.
* Balances classes with SMOTE.
* Trains Logistic Regression.
* Evaluates and saves model.

### app.py

* User-friendly interface using Streamlit.
* Integrates scraper, model prediction, and sentiment.
* Displays insightful summaries and suggestions.

---

## Requirements

* Python 3.8+
* pandas
* scikit-learn
* imbalanced-learn
* selenium
* streamlit
* transformers
* torch
* textblob

Install requirements via:

```bash
pip install -r requirements.txt
```

---

## Notes

* ChromeDriver path must be set correctly in `Scraper.py`.
* The pretrained models require internet access to download on first run.
* Scraping Amazon may be subject to their terms of service; use responsibly.
* Sentiment scores are approximate and based on the pretrained sentiment model.

---


