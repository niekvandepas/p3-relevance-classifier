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

DATA_PATH = os.getenv("REDDIT_DATA_PATH")

if not DATA_PATH:
    raise ValueError(
        "Please set the REDDIT_DATA_PATH environment variable in your .env file."
    )

# ==========================================
# 1. Load and Prepare the Data
# ==========================================

print_header("Parsing JSON")

raw_texts = []
pool_labels = []

total_lines = 16712355 # Output of wc -l

with open(DATA_PATH, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if i % 100000 == 0:
            print(f"Processed {i} out of 16,712,355 lines ({(i/total_lines)*100:.2f}%)")
        item = json.loads(line)

        text = ""
        # If item is a comment, include it
        if "body" in item:
            text = item["body"]

        # If item is a submission, include it only if it has self text (e.g. skip link- or image-posts, which can't be feasibly labeled or analyzed).
        elif "title" in item:
            if item.get("selftext", "").strip():
                text = item["title"] + " " + item["selftext"]
            else:
                continue

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
# 3. Initialize or Resume with Seed
# ==========================================
np.random.seed(42)

ANNOTATIONS_FILE = "reddit_annotations_progress.json"
annotations_dict = {}

print_header("Initializing or Resuming Labels")

# Check if there is previous progress to load
if os.path.exists(ANNOTATIONS_FILE):
    print(f"Found existing progress in {ANNOTATIONS_FILE}. Resuming...")
    with open(ANNOTATIONS_FILE, "r") as f:
        annotations_dict = json.load(f)

    # Convert string keys from JSON back to integers
    resumed_indices = np.array([int(k) for k in annotations_dict.keys()])
    resumed_labels = np.array(list(annotations_dict.values()))

    active_learner.initialize_data(resumed_indices, resumed_labels)

else:
    keywords = ["eten", "culinair", "kaas", "stamppot", "recept", "stroopwafel"]
    pattern = re.compile(r"\b(" + "|".join(keywords) + r")\b", re.IGNORECASE)
    candidate_indices = [
        idx for idx, text in enumerate(raw_texts) if pattern.search(text)
    ]

    print(
        f"Found {len(candidate_indices)} potential hits out of {len(raw_texts)} total texts using keyword search."
    )

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

    for idx, lbl in zip(initial_indices, seed_labels):
        annotations_dict[str(idx)] = int(lbl)

    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)


# ==========================================
# 4. The Active Learning loop
# ==========================================
samples_per_query = 100

print_header(f"Starting labeling session (Batch size: {samples_per_query})...\n")

queried_indices = active_learner.query(num_samples=samples_per_query)

current_labels = []
quit_requested = False

for count, idx in enumerate(queried_indices):
    print(f"\nItem {count+1}/{samples_per_query}")
    print(preview_text(raw_texts[idx]))
    print("")

    label = input("Label (0=No, 1=Yes, 'q'=Save and Quit): ")

    if label.lower() == "q":
        quit_requested = True
        break

    if label not in ["0", "1"]:
        raise ValueError("Invalid label! Please enter 0, 1, or q.")

    current_labels.append(int(label))

    print("\n\n\n\n\n")
    print_divider()

# ===========================================
# 5. Handle model updates and progress saving
# ===========================================

if quit_requested:
    # We quit mid-batch. Save the partial labels to JSON, but DO NOT call active_learner.update()
    # because passing fewer labels than queried will cause a small-text shape mismatch error. We'll just pick them up next time.
    processed_indices = queried_indices[: len(current_labels)]
    for idx, lbl in zip(processed_indices, current_labels):
        annotations_dict[str(idx)] = int(lbl)

    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)

    print_header("Early exit requested. Partial batch saved to JSON. Wrapping up...")

else:
    # Full batch completed. Update the model and save to JSON.
    active_learner.update(np.array(current_labels))

    for idx, lbl in zip(queried_indices, current_labels):
        annotations_dict[str(idx)] = int(lbl)

    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)

print(f"Total labeled: {len(active_learner.indices_labeled)}")

print("\nSaving the model and vectorizer...")

final_model = active_learner.classifier.model  # type: ignore

joblib.dump(final_model, "relevance_model.pkl")
joblib.dump(vectorizer, "relevance_vectorizer.pkl")

print("Saved successfully!")
