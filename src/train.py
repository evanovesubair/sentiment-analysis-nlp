"""Sentiment analysis of short reviews with TF-IDF and scikit-learn.

Covers: text cleaning, TF-IDF features, Logistic Regression and Naive Bayes,
stratified cross-validation, a held-out test set, and a prediction helper.

Run:
    python src/make_dataset.py                # build data/synthetic_reviews.csv (once)
    python src/train.py                       # train, then also test on hand-written reviews
    python src/train.py --data my_reviews.csv # any CSV with 'text' and 'label' columns
    python src/train.py --predict "Great phone, loved it"
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "sentiment_model.joblib"
EXTERNAL_PATH = ROOT / "data" / "hand_written_reviews.csv"


def clean(text: str) -> str:
    """Lowercase, drop URLs and punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_pipeline(classifier) -> Pipeline:
    # No stop-word removal: words like "not" and "no" carry sentiment.
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("clf", classifier),
        ]
    )


def predict(text: str):
    model = joblib.load(MODEL_PATH)
    label = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    conf = float(max(proba))
    print(f"{label} (confidence {conf:.2f})")


def train(data_path: Path):
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(data_path).dropna(subset=["text", "label"])
    print("Samples:", len(df))
    print(df["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED
    )

    candidates = {
        "Logistic Regression": (
            make_pipeline(LogisticRegression(max_iter=2000, random_state=SEED)),
            {"clf__C": [0.1, 1, 10, 100]},
        ),
        "Naive Bayes": (
            make_pipeline(MultinomialNB()),
            {"clf__alpha": [0.1, 0.5, 1.0]},
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results, fitted = {}, {}
    for name, (pipe, grid) in candidates.items():
        baseline = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy")
        search.fit(X_train, y_train)
        best = search.best_estimator_
        pred = best.predict(X_test)
        results[name] = {
            "baseline_cv_accuracy": round(float(baseline.mean()), 4),
            "best_params": search.best_params_,
            "tuned_cv_accuracy": round(float(search.best_score_), 4),
            "test_accuracy": round(accuracy_score(y_test, pred), 4),
            "test_precision": round(precision_score(y_test, pred, pos_label="positive"), 4),
            "test_recall": round(recall_score(y_test, pred, pos_label="positive"), 4),
            "test_f1": round(f1_score(y_test, pred, pos_label="positive"), 4),
            "test_samples": int(len(y_test)),
        }
        fitted[name] = best
        print(f"\n{name}\n{json.dumps(results[name], indent=2)}")
        print(classification_report(y_test, pred))

        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, colorbar=False)
        ax.set_title(f"Confusion matrix: {name}")
        fig.tight_layout()
        fig.savefig(OUT / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    # Most influential words, from the Logistic Regression model
    lr = fitted["Logistic Regression"]
    vocab = lr.named_steps["tfidf"].get_feature_names_out()
    coefs = pd.Series(lr.named_steps["clf"].coef_[0], index=vocab).sort_values()
    # classes_ are alphabetical: negative, positive -> positive weights favour 'positive'
    top = pd.concat([coefs.head(10), coefs.tail(10)])
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:red"] * 10 + ["tab:green"] * 10
    top.plot.barh(ax=ax, color=colors)
    ax.set_title("Most influential words (Logistic Regression)")
    ax.set_xlabel("Weight (negative = pushes to 'negative', positive = pushes to 'positive')")
    fig.tight_layout()
    fig.savefig(OUT / "top_words.png", dpi=150)
    plt.close(fig)

    # Out-of-distribution check: reviews written by hand, never seen in training.
    # This is a more honest guide to real-world behaviour than the in-distribution split.
    if EXTERNAL_PATH.exists() and EXTERNAL_PATH.resolve() != data_path.resolve():
        ext = pd.read_csv(EXTERNAL_PATH).dropna(subset=["text", "label"])
        for name, model in fitted.items():
            ext_pred = model.predict(ext["text"])
            results[name]["external_accuracy"] = round(accuracy_score(ext["label"], ext_pred), 4)
            results[name]["external_samples"] = int(len(ext))
            print(f"{name}: accuracy on {len(ext)} hand-written reviews = "
                  f"{results[name]['external_accuracy']}")
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(ext["label"], ext_pred, ax=ax, colorbar=False)
            ax.set_title(f"Hand-written reviews: {name}")
            fig.tight_layout()
            fig.savefig(OUT / f"external_confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
            plt.close(fig)

    best_name = max(results, key=lambda n: results[n]["tuned_cv_accuracy"])
    print(f"\nSaving '{best_name}' (highest cross-validated accuracy) to {MODEL_PATH}")
    joblib.dump(fitted[best_name], MODEL_PATH)
    results["saved_model"] = best_name
    with open(OUT / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "synthetic_reviews.csv")
    parser.add_argument("--predict", type=str, help="Classify a piece of text with the saved model")
    args = parser.parse_args()
    if args.predict:
        predict(args.predict)
    else:
        train(args.data)
