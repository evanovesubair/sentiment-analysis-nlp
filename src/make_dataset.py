"""Build a synthetic review dataset for training.

IMPORTANT: this data is generated from templates, so it is much simpler than real
reviews. It exists so the project runs end to end with no download. For a realistic
model, train on a real dataset (e.g. IMDB or Amazon reviews) with:

    python src/train.py --data path/to/real_reviews.csv

Run:  python src/make_dataset.py
"""
import random
from pathlib import Path

import pandas as pd

SEED = 42
N_PER_CLASS = 300
OUT = Path(__file__).resolve().parent.parent / "data" / "synthetic_reviews.csv"

SUBJECTS = [
    "The movie", "The phone", "The food", "The hotel", "The laptop", "The service",
    "The app", "The book", "The headphones", "The restaurant", "The delivery",
    "The course", "The jacket", "The game", "The concert", "The sofa", "The camera",
    "The tablet", "The tour", "The coffee",
]
POSITIVE = [
    "was excellent", "was fantastic", "was wonderful", "is amazing", "is really good",
    "was brilliant", "is superb", "was a delight", "exceeded my expectations",
    "is worth every penny", "is great value", "works perfectly", "is lovely",
    "was outstanding", "was impressive and enjoyable",
]
NEGATIVE = [
    "was terrible", "was awful", "was disappointing", "is really bad",
    "was a waste of money", "is poor quality", "was boring and slow", "is useless",
    "fell far short of my expectations", "is not worth the price", "broke after a week",
    "works badly", "was frustrating and annoying", "is horrible", "was a complete letdown",
]
# Negation flips the meaning, so these make word order and bigrams matter.
NEGATED_POSITIVE = ["was not good", "is not worth it", "was not enjoyable", "is not great",
                    "did not work well", "was not impressive"]
NEGATED_NEGATIVE = ["was not bad at all", "is not disappointing", "was not boring",
                    "is not overpriced", "did not let me down", "was not a waste"]
OPENERS = ["", "", "Honestly, ", "To be fair, ", "I think ", "In my opinion, ", "Overall, "]
POS_CLOSERS = ["", "", "I would buy it again.", "Highly recommended.", "Very happy with it."]
NEG_CLOSERS = ["", "", "I would not buy it again.", "Avoid it.", "Very unhappy with it."]


def make_review(rng: random.Random, positive: bool) -> str:
    if positive:
        pool = POSITIVE if rng.random() < 0.8 else NEGATED_NEGATIVE
        closer = rng.choice(POS_CLOSERS)
    else:
        pool = NEGATIVE if rng.random() < 0.8 else NEGATED_POSITIVE
        closer = rng.choice(NEG_CLOSERS)
    opener = rng.choice(OPENERS)
    subject = rng.choice(SUBJECTS)
    if opener:
        subject = subject[0].lower() + subject[1:]
    sentence = f"{opener}{subject} {rng.choice(pool)}."
    return f"{sentence} {closer}".strip()


def main():
    rng = random.Random(SEED)
    rows = []
    for label in ("positive", "negative"):
        seen = set()
        while len(seen) < N_PER_CLASS:
            seen.add(make_review(rng, label == "positive"))
        rows += [(text, label) for text in seen]
    df = pd.DataFrame(rows, columns=["text", "label"]).sample(frac=1, random_state=SEED)
    # A review text must belong to exactly one class
    df = df.drop_duplicates(subset="text", keep=False)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
