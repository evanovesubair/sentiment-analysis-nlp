# Sentiment Analysis of Reviews (NLP)

A small natural language processing project that classifies short product, movie and restaurant reviews as **positive** or **negative** using TF-IDF features and scikit-learn.

It demonstrates the basic NLP workflow: text cleaning, vectorisation with word and bigram TF-IDF, model comparison (Logistic Regression vs Naive Bayes), cross-validation, hyperparameter tuning, and an honest evaluation on text the model has not seen.

## Data

| File | What it is |
|---|---|
| `data/synthetic_reviews.csv` | 600 reviews generated from templates by `src/make_dataset.py` (used for training) |
| `data/hand_written_reviews.csv` | 122 reviews I wrote by hand (used only as an external test set) |

The training data is **synthetic**: short templated sentences, so it is far simpler than real reviews. That keeps the project runnable with no download, but it also means the training scores below are not a measure of real-world quality.

## Approach

1. **Clean** text: lowercase, strip URLs and punctuation. Stop words are kept because words like "not" and "no" carry sentiment.
2. **Vectorise** with `TfidfVectorizer` using unigrams and bigrams.
3. **Train** Logistic Regression and Multinomial Naive Bayes in scikit-learn pipelines.
4. **Tune** with `GridSearchCV` and 5-fold stratified cross-validation.
5. **Evaluate** on a held-out 20% split and on the hand-written reviews.

## Results

| Model | CV accuracy | Held-out test accuracy (synthetic, 120 reviews) | Hand-written reviews (122) |
|---|---|---|---|
| Logistic Regression (`C=100`) | 0.994 | 1.000 | 0.639 |
| Naive Bayes (`alpha=0.1`) | 0.996 | 1.000 | 0.639 |

The perfect scores on the synthetic split are not meaningful: the templates repeat the same small vocabulary, so the models can memorise it. The **64% on hand-written reviews is the realistic number**, and it shows that a model trained on this synthetic data does not generalise well to natural wording.

## Known limitations

- **Poor generalisation.** Hand-written reviews use words and phrasings the synthetic data never contained ("battery dies by lunch", "grainy and slow").
- **Negation is fragile.** For example, "This was not good at all" is classified as positive, because the phrase "at all" appears in the synthetic positive template "not bad at all". This is a good example of a model learning a spurious pattern.
- **Only two classes, English only, short text only.**

A first attempt that trained on the hand-written reviews alone scored below chance in cross-validation. Each negative review mirrored a positive one, so topic words appeared once in each class, which misleads the model when one of them is held out. That is why the hand-written set is used only for testing.

## Run it

```bash
git clone <your-repo-url>
cd sentiment-analysis-nlp
pip install -r requirements.txt

python src/make_dataset.py        # regenerate the synthetic training data
python src/train.py               # train, evaluate, save the best model
python src/train.py --predict "The food was delicious and the staff were lovely"
```

### Use a real dataset (recommended next step)

Any CSV with `text` and `label` columns (labels `positive` / `negative`) works. For example, after downloading the IMDB reviews dataset and converting it to that format:

```bash
python src/train.py --data data/imdb_reviews.csv
```

With real data the held-out score will be lower than 100% but far more trustworthy.

## Project structure

```
sentiment-analysis-nlp/
├── data/
│   ├── synthetic_reviews.csv
│   └── hand_written_reviews.csv
├── src/
│   ├── make_dataset.py   # builds the synthetic training data
│   └── train.py          # train, tune, evaluate, predict
├── outputs/              # metrics.json, confusion matrices, top-words chart
├── requirements.txt
└── README.md
```

## Possible next steps

- Train on a real dataset such as IMDB and report the honest score.
- Add a neutral class and handle mixed reviews.
- Try a pretrained transformer model and compare it with the TF-IDF baseline.
