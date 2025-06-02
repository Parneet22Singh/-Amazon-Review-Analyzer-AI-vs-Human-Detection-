# Amazon Review Analyzer

This project is an AI-powered tool that scrapes Amazon product reviews and analyzes them to detect whether reviews are written by real humans or AI-generated bots. It also performs sentiment analysis and provides a purchase suggestion based on the review quality.

---

## Features

- **Web Scraping:** Uses Selenium to scrape multiple pages of Amazon product reviews automatically.
- **AI vs Human Detection:** A Logistic Regression model trained on a labeled dataset to classify reviews as AI-generated or human-written.
- **Sentiment Analysis:** Calculates sentiment polarity for each review to assess overall positivity or negativity.
- **Summary Report:** Displays counts and percentages of AI vs human reviews, average sentiment score.
- **Purchase Suggestion:** Provides user-friendly advice based on AI presence and sentiment analysis.
- **Interactive Web UI:** Built using Streamlit for easy input and real-time results.

---

## Project Structure

- `Train_model.py`:  
  Loads labeled review data, cleans and vectorizes text, balances classes using SMOTE, trains a Logistic Regression model, evaluates performance, and saves the model and vectorizer.

- `Scraper.py`:  
  Uses Selenium to scrape Amazon product reviews from a given URL, handling multiple pages and saving the collected reviews to a CSV file.

- `App.py`:  
  Streamlit app that takes a product review URL input, scrapes reviews, predicts AI vs human labels, performs sentiment analysis, and displays results and purchase suggestions.

- `dataset/`:  
  Contains datasets, trained model (`trained.pkl`), and vectorizer (`vectorizer.pkl`).

- `preprocess.py`:  
  (Assumed) Contains text cleaning and vectorization helper functions.

---

## Data Labeling Process

The labeled dataset used to train the AI vs Human review classification model is prepared separately. After scraping raw reviews, a labeling script is run to assign labels indicating whether a review is AI-generated or human-written. This labeling also includes sentiment analysis to understand how positive or negative the reviews are, helping assess the overall product quality.

---

## Requirements

- Python 3.7+
- Packages:
  - pandas
  - scikit-learn
  - imblearn
  - selenium
  - joblib
  - streamlit
  - textblob

- Chrome browser and matching ChromeDriver executable (place in project root)

---

## Installation & Setup

1. Clone the repository:
   ```bash
   git clone <repo_url>
   cd <repo_folder>
