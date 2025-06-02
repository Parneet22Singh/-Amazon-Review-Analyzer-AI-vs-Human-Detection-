import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from preprocess import clean_text, vectorize_text

# Load the dataset
df = pd.read_csv("dataset/labelled_reviews_with_ai_detection.csv")

# Drop labels with less than 2 samples
label_counts = df["label"].value_counts()
df = df[df["label"].isin(label_counts[label_counts >= 2].index)]

# Check if 'label' column exists
if "label" not in df.columns:
    print("❌ Error: 'label' column not found in the dataset.")
    exit()

# Clean the review text
df["cleaned"] = df["Review"].apply(lambda x: clean_text(x) if isinstance(x, str) else "")

# Re-check label distribution
label_counts = df["label"].value_counts()
print("Label Distribution:\n", label_counts)

# Abort if any class still has fewer than 2 samples (safety check)
if any(label_counts < 2):
    print("\n❌ Training aborted: At least one class has fewer than 2 samples.")
    exit()

# Vectorize the cleaned text
vectorizer, vectors = vectorize_text(df["cleaned"])

# Split data (80% train, 20% test) with stratified labels
X_train, X_test, y_train, y_test = train_test_split(
    vectors, df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# Balance the training data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train a Logistic Regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train_resampled, y_train_resampled)

# Evaluate on test set
y_pred = model.predict(X_test)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Save model and vectorizer
joblib.dump(model, "dataset/trained.pkl")
joblib.dump(vectorizer, "dataset/vectorizer.pkl")

print("\n✅ Model and vectorizer saved successfully.")
