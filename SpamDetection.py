import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

df = pd.read_csv("D:\\Downloads\\sms_spam_200.csv")

X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Naive Bayes": Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english")),
        ("model", MultinomialNB())
    ]),

    "Logistic Regression": Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english")),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "Linear SVM": Pipeline([
        ("vectorizer", CountVectorizer(stop_words="english")),
        ("model", LinearSVC())
    ])
}

best_accuracy = 0
best_model = None
best_model_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name


joblib.dump(best_model, "best_model.pkl")

print("\n✅ Best Model Selected:", best_model_name)
print("✅ Best Accuracy:", best_accuracy)
print("✅ Saved as best_model.pkl")
