import json
import re
import shutil
import textwrap

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# small-text specific imports
from small_text import (
    LABEL_UNLABELED,
    PoolBasedActiveLearner,
    SklearnDataset,
    random_initialization,
)
from small_text.classifiers import SklearnClassifierFactory
from small_text.query_strategies import LeastConfidence

import joblib
from dotenv import load_dotenv
import os

TEXT_PREVIEW_LENGTH = 2000


def print_header(text: str) -> None:
    """
    Print the given string prepended by a divider for better visibility.
    """
    print(f"=====> {text}")


def print_divider() -> None:
    """
    Print a divider across the entire width of the terminal window.
    """
    width = shutil.get_terminal_size().columns
    print("-" * width)


def preview_text(text: str, margin_lines: int = 5) -> str:
    cols, rows = shutil.get_terminal_size()

    usable_rows = max(rows - margin_lines, 1)
    max_chars = cols * usable_rows

    clipped = text[:max_chars]

    # Wrap nicely so it respects terminal width
    return "\n".join(textwrap.wrap(clipped, width=cols))


load_dotenv()

DATA_PATH = os.getenv("DELPHER_DATA_PATH")

if not DATA_PATH:
    raise ValueError(
        "Please set the DELPHER_DATA_PATH environment variable in your .env file."
    )

# ==========================================
# 1. Load and Prepare the Data
# ==========================================

print_header("Parsing JSON")

raw_texts = []
pool_labels = []

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)

        if item["title"] == "Advertentie":
            continue

        # 'plain_text' contains the title, so no need to concatenate them.
        text = item["plain_text"]

        # Only consider non-empty texts
        if text.strip():
            raw_texts.append(text)
            pool_labels.append(LABEL_UNLABELED)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(max_features=10000)
x_features = vectorizer.fit_transform(raw_texts)

y_labels = np.array(pool_labels)

# Wrap the data in small-text's specific SklearnDataset format
target_labels = np.array([0, 1])
train_dataset = SklearnDataset(x_features, y_labels, target_labels=target_labels)


# ==========================================
# 2. Set up the Active Learner
# ==========================================

classifier_factory = SklearnClassifierFactory(MultinomialNB(), num_classes=2)

# 'LeastConfidence' mathematically picks the items the model is most unsure about
query_strategy = LeastConfidence()

active_learner = PoolBasedActiveLearner(
    classifier_factory, query_strategy, train_dataset
)


# ==========================================
# 3. Initialize with a guided seed
# ==========================================
np.random.seed(42)

print_header("Initializing seed labels")

keywords = ["eten", "culinair", "kaas", "stamppot", "recept", "stroopwafel"]

# Use a word boundary regex pattern rather than plain text search, since "eten" would otherwise include "weten", "meten", etc.
pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)
candidate_indices = [idx for idx, text in enumerate(raw_texts) if pattern.search(text)]

print(
    f"Found {len(candidate_indices)} potential hits out of {len(raw_texts)} total texts using keyword search."
)


# Randomly select 5 samples from this concentrated list, falling back to random if not enough hits
if len(candidate_indices) >= 5:
    initial_indices = np.random.choice(candidate_indices, size=5, replace=False)
else:
    print("Not enough keyword hits! Falling back to random initialization.")
    initial_indices = random_initialization(train_dataset, n_samples=5)

seed_labels = []

for count, idx in enumerate(initial_indices):
    print(f"\nSeed Item {count+1}/5")
    print(preview_text(raw_texts[idx]))
    print("")

    label = input("Label (0 for Not Relevant, 1 for Relevant (Dutch Cuisine)): ")
    if label not in ["0", "1"]:
        raise ValueError("Invalid label! Please enter 0 or 1.")
    seed_labels.append(int(label))

    print("\n\n\n\n\n")
    print_divider()

active_learner.initialize_data(initial_indices, np.array(seed_labels))


# ==========================================
# 4. The Active Learning loop
# ==========================================
num_queries = 3
samples_per_query = 5

print_header("Starting Active Learning Loop...\n")

for i in range(num_queries):
    # Step A: The learner calculates uncertainties and asks for the most informative texts
    queried_indices = active_learner.query(num_samples=samples_per_query)

    # Step B: manual labeling
    current_labels = []
    for count, idx in enumerate(queried_indices):
        print(f"\nItem {count+1}/{samples_per_query}")
        print(preview_text(raw_texts[idx]))
        print("")

        label = input("Label (0 for Not Relevant, 1 for Relevant (Dutch Cuisine)): ")
        if label not in ["0", "1"]:
            raise ValueError("Invalid label! Please enter 0 or 1.")

        current_labels.append(int(label))

        print("\n\n\n\n\n")
        print_divider()

    # Step C: Feed the answers back to the model so it can retrain itself
    active_learner.update(np.array(current_labels))

    print(f"Iteration {i+1} | Total labeled: {len(active_learner.indices_labeled)}")

print("\nSaving the model and vectorizer...")

final_model = active_learner.classifier.model

# Both the model and the vectorizer need to be saved for reproducibility
joblib.dump(final_model, "relevance_model.pkl")
joblib.dump(vectorizer, "relevance_vectorizer.pkl")

print("Saved successfully!")
