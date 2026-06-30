import json
from pathlib import Path
import re
import time
import warnings

import numpy as np
from nltk.corpus import stopwords

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
from huggingface_hub import hf_hub_download

from constants import (
    REDDIT_ANNOTATIONS_FILE,
    REDDIT_CACHE_FILE,
    REDDIT_LANGUAGE,
    REDDIT_MODEL_FILE,
    REDDIT_VECTORIZER_FILE,
)
from util import preview_text, print_divider, print_header

TEXT_PREVIEW_LENGTH = 2000

# The language of the dataset to use. The dataset is split into Dutch and English-language items,
# and classifying both languages with a single model is not ideal, so we classify seperately.
# This variable is used to set the appropriate stop word list and to import the correct dataset.
# It is also appended to the model and vectorizer filenames when saving,
# to avoid collisions if you run the script multiple times with different languages.
LANGUAGE = REDDIT_LANGUAGE


REDDIT_STOP_WORDS = (
    list(set(stopwords.words("dutch")).union({"mijn", "ik", "zijn", "was", "we"}))
    if REDDIT_LANGUAGE == "nl"
    else list(set(stopwords.words("english")))
)

KEYWORDS = (
    [
        "eten",
        "culinair",
        "nederlands",
        "hollands",
        "stamppot",
        "stroopwafel",
        "bitterballen",
        "boerenkool",
        "erwtensoep",
        "kroket",
        "poffertjes",
        "pannenkoeken",
        "haring",
        "kibbeling",
        "drop",
        "hagelslag",
        "smakelijk",
        "gerecht",
        "lekker",
        "maaltijd",
    ]
    if LANGUAGE == "nl"
    else [
        "food",
        "culinary",
        "dutch",
        "stamppot",
        "stroopwafel",
        "stroopwafels",
        "bitterballen",
        "bitterbal",
        "poffertjes",
        "kroket",
        "kibbeling",
        "hagelslag",
        "kale",
        "pea soup",
        "croquette",
        "croquettes",
        "pancakes",
        "herring",
        "licorice",
        "chocolate sprinkles",
        "dish",
        "tasty",
        "meal",
    ]
)

KEYWORDS_PATTERN = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b", re.IGNORECASE)

load_dotenv()

HF_REPO_ID = os.environ.get("HF_REPO_ID", "YOUR_HF_REPO_ID")  # Set this in your .env
HF_TOKEN = os.environ.get("HF_TOKEN")  # Set this in your .env


def get_data_path(file_type: str, language: str) -> Path:
    """
    Returns a Path to the local cached version of the HuggingFace file.
    file_type: 'posts' or 'comments'
    language: 'en' or 'nl'
    """
    filename = f"reddit-{language}-{file_type}.ndjson"

    # This will download the file if missing, or return the path if it exists
    cached_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename, repo_type="dataset", token=HF_TOKEN
    )

    return Path(cached_path)


# Ensure directories exist before trying to read/write files
Path("artifacts/models").mkdir(parents=True, exist_ok=True)
Path("artifacts/cache").mkdir(parents=True, exist_ok=True)
Path("annotations").mkdir(parents=True, exist_ok=True)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ==========================================
# 1. Load and Prepare the Data
# ==========================================

if os.path.exists(REDDIT_CACHE_FILE):
    start_time = time.time()

    print_header("Loading cached texts and features from disk...")
    cached_data = joblib.load(REDDIT_CACHE_FILE)

    item_ids: list[str] = cached_data["item_ids"]
    raw_texts: list[str] = cached_data["raw_texts"]
    pool_labels = cached_data["pool_labels"]

    x_features = cached_data["x_features"]
    vectorizer = cached_data["vectorizer"]

    y_labels = np.array(pool_labels)

    print(f"Loaded cached data in {time.time() - start_time:.2f} seconds!")

else:
    print_header("Downloading and Parsing HF JSON Data")

    item_ids: list[str] = []
    raw_texts: list[str] = []
    pool_labels: list[int] = []

    # Loop through both posts and comments files
    for file_type in ["posts", "comments"]:
        print(f"Processing {file_type}...")
        data_path = get_data_path(file_type, LANGUAGE)

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text", "")

                # Only consider non-empty texts
                if text.strip():
                    # Check that the text has meat on its bones (e.g., at least 25-30 words).
                    # This completely excludes 1-sentence comments and title-only submissions.
                    if len(text.split()) < 100:
                        continue

                    item_ids.append(item["id"])
                    raw_texts.append(text)
                    pool_labels.append(LABEL_UNLABELED)

    print(f"Total items parsed: {len(raw_texts)}")

    print_header("Computing TF-IDF")

    vectorizer = TfidfVectorizer(
        max_features=10000, stop_words=REDDIT_STOP_WORDS, max_df=0.5, min_df=5
    )
    x_features = vectorizer.fit_transform(raw_texts)

    y_labels = np.array(pool_labels)

    print_header("Saving parsed data and vectorizer to cache...")
    joblib.dump(
        {
            "item_ids": item_ids,
            "raw_texts": raw_texts,
            "pool_labels": pool_labels,
            "x_features": x_features,
            "vectorizer": vectorizer,
        },
        REDDIT_CACHE_FILE,
    )

# Wrap the data in small-text's specific SklearnDataset format
target_labels = np.array([0, 1])

train_dataset = SklearnDataset(x_features, y_labels, target_labels=target_labels)

id_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}

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

annotations_dict = {}

print_header("Initializing or Resuming Labels")

# Check if there is previous progress to load
if os.path.exists(REDDIT_ANNOTATIONS_FILE):
    print(f"Found existing progress in {REDDIT_ANNOTATIONS_FILE}. Resuming...")
    with open(REDDIT_ANNOTATIONS_FILE, "r") as f:
        annotations_dict = json.load(f)

    resumed_indices = []
    resumed_labels = []

    for saved_id, label in annotations_dict.items():
        if saved_id in id_to_index:
            resumed_indices.append(id_to_index[saved_id])
            resumed_labels.append(label)
        else:
            print(
                f"Warning: Cached ID {saved_id} not found in the current dataset. Skipping."
            )

    active_learner.initialize_data(np.array(resumed_indices), np.array(resumed_labels))

else:
    # Filter for items that contain at least four of the above keywords.
    # This is a number that was experimentally determined:
    # too high and you get very few hits; too low and you get pointless seed items.
    # It doesn't really matter that much anyway, since this is only to get some useful seed labels.
    # The real classification is done later, after labeling.
    # The items should also be relatively long, to ensure they contain enough information to label.
    # This is because the dataset contains a lot of very short comments that are not useful for training,
    # which confuses the model because it will over-weight certain words that appear in those short comments but are not actually relevant to the task.
    print(
        "No existing progress found. Starting fresh initialization with keyword filtering..."
    )

    candidate_indices = []

    for idx, text in enumerate(raw_texts):
        print(
            f"Scanning item {idx+1}/{len(raw_texts)} for keyword matches...",
            end="\r",
        )

        unique_matches = set(m.lower() for m in KEYWORDS_PATTERN.findall(text))
        unique_match_count = len(unique_matches)

        if unique_match_count >= 4 and len(text) >= 150:
            candidate_indices.append(idx)

    print(
        f"\nFound {len(candidate_indices)} potential hits out of {len(raw_texts)} total texts using keyword search."
    )

    if len(candidate_indices) >= 25:
        initial_indices = np.random.choice(candidate_indices, size=25, replace=False)
    else:
        print("Not enough keyword hits! Falling back to random initialization.")
        initial_indices = random_initialization(train_dataset, n_samples=25)

    seed_labels = []

    for count, idx in enumerate(initial_indices):
        print(f"\nSeed Item {count+1}/25 (ID: {item_ids[idx]})")
        print(preview_text(raw_texts[idx]))
        print("")

        while True:
            label = input(
                "Label (0 for Not Relevant, 1 for Relevant (Dutch culinary culture)): "
            ).strip()

            if label in ["0", "1"]:
                break

            print("Enter 0 or 1")
        seed_labels.append(int(label))

        print("\n\n\n\n\n")
        print_divider()

    active_learner.initialize_data(initial_indices, np.array(seed_labels))

    for idx, lbl in zip(initial_indices, seed_labels):
        item_id = item_ids[idx]
        annotations_dict[item_id] = int(lbl)

    with open(REDDIT_ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)


# ==========================================
# 4. The Active Learning loop
# ==========================================
samples_per_query = 25

print_header(f"Starting labeling session (Batch size: {samples_per_query})...\n")

# Step A: The learner calculates uncertainties and asks for the most informative texts
queried_indices = active_learner.query(num_samples=samples_per_query)

# Step B: manual labeling
current_labels = []
quit_requested = False

for count, idx in enumerate(queried_indices):
    matches = set(m.lower() for m in KEYWORDS_PATTERN.findall(raw_texts[idx]))
    keywords_count = len(matches)

    print(
        f"\nItem {count+1}/{samples_per_query}. {keywords_count}/{len(KEYWORDS)} keywords. (ID: {item_ids[idx]})"
    )

    print(preview_text(raw_texts[idx]))
    print("")

    while True:
        label = input("Label (0=No, 1=Yes, 'q'=Save and Quit): ").strip()

        if label in ["0", "1", "q"]:
            break

        print("Enter 0 or 1")

    if label.lower() == "q":
        quit_requested = True
        break

    current_labels.append(int(label))

    print("\n\n\n\n\n\n\n\n")
    print_divider()

# ===========================================
# 5. Handle model updates and progress saving
# ===========================================

if quit_requested:
    processed_indices = queried_indices[: len(current_labels)]

    for idx, lbl in zip(processed_indices, current_labels):
        item_id = item_ids[idx]
        annotations_dict[item_id] = int(lbl)

    with open(REDDIT_ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)

    print_header("Early exit requested. Partial batch saved to JSON. Wrapping up...")

else:
    active_learner.update(np.array(current_labels))

    for idx, lbl in zip(queried_indices, current_labels):
        item_id = item_ids[idx]
        annotations_dict[item_id] = int(lbl)

    with open(REDDIT_ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)

print(f"Total labeled: {len(annotations_dict)}")

print("\nSaving the model and vectorizer...")

final_model = active_learner.classifier.model  # type: ignore

joblib.dump(final_model, REDDIT_MODEL_FILE)
joblib.dump(vectorizer, REDDIT_VECTORIZER_FILE)

print("Saved successfully!")
