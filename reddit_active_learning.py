import json
from pathlib import Path
import re
import time
import warnings

import numpy as np

from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

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

from constants import (
    REDDIT_ANNOTATIONS_FILE,
    REDDIT_CACHE_FILE,
    REDDIT_LANGUAGE,
    REDDIT_MODEL_FILE,
)


TEXT_PREVIEW_LENGTH = 2000

# The language of the dataset to use. The dataset is split into Dutch and English-language items,
# and classifying both languages with a single model is not ideal, so we classify seperately.
# This variable is used to set the appropriate stop word list and to import the correct dataset.
# It is also appended to the model and vectorizer filenames when saving,
# to avoid collisions if you run the script multiple times with different languages.
LANGUAGE = REDDIT_LANGUAGE

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

# KEYWORDS_PATTERN = re.compile(r"\b(" + "|".join(KEYWORDS) + r")\b", re.IGNORECASE)
KEYWORDS_PATTERN = re.compile(r"De Nederlandse keuken", re.IGNORECASE)

load_dotenv()

REDDIT_DATA_FOLDER = os.getenv("REDDIT_DATA_FOLDER")

if not REDDIT_DATA_FOLDER:
    raise ValueError(
        "Please set the REDDIT_DATA_FOLDER environment variable in your .env file."
    )

REDDIT_DATA_FILENAME = f"reddit-{REDDIT_LANGUAGE}.ndjson"

REDDIT_DATA_PATH = Path(REDDIT_DATA_FOLDER) / REDDIT_DATA_FILENAME


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

    print_header("Loading cached texts and embeddings from disk...")
    cached_data = joblib.load(
        REDDIT_CACHE_FILE,
        mmap_mode="c", # Read embeddings from disk rather than dumping 30GB into RAM.
    )
    raw_texts: list[str] = cached_data["raw_texts"]
    pool_labels = cached_data["pool_labels"]
    x_embeddings = cached_data["x_embeddings"]
    y_labels = np.array(pool_labels)

    print(f"Loaded cached data in {time.time() - start_time:.2f} seconds!")

else:
    print_header("Parsing JSON")

    raw_texts: list[str] = []
    pool_labels: list[int] = []

    total_lines = 10651836 if LANGUAGE == "nl" else 4257108  # Output of wc -l

    with open(REDDIT_DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i % 100000 == 0:
                print(
                    f"Processed {i} out of {format(total_lines, ',')} lines ({i/total_lines:.1%})..."
                )
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

    print_header("Computing RoBBERT Embeddings (This will take a while!)")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    robbert_model = SentenceTransformer(
        "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers",
        device=device,
    )

    BATCH_SIZE = 5000
    embeddings_list: list[torch.Tensor] = []

    for i in tqdm(range(0, len(raw_texts), BATCH_SIZE), desc="Encoding batches"):
        batch = raw_texts[i : i + BATCH_SIZE]
        batch_emb = robbert_model.encode(batch, show_progress_bar=False)
        embeddings_list.append(batch_emb)

    # Stack the list of batches into a single numpy array
    x_embeddings = np.vstack(embeddings_list)

    y_labels = np.array(pool_labels)

    print_header("Saving parsed data and embeddings to cache...")
    joblib.dump(
        {
            "raw_texts": raw_texts,
            "pool_labels": pool_labels,
            "x_embeddings": x_embeddings,
        },
        REDDIT_CACHE_FILE,
    )

# Wrap the data in small-text's specific SklearnDataset format
target_labels = np.array([0, 1])
train_dataset = SklearnDataset(x_embeddings, y_labels, target_labels=target_labels)


# ==========================================
# 2. Set up the Active Learner
# ==========================================

lgbm_model = LGBMClassifier(
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=200,
    class_weight="balanced",
    n_jobs=-1,
    verbosity=-1,
)
classifier_factory = SklearnClassifierFactory(lgbm_model, num_classes=2)

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

    # Convert string keys from JSON back to integers
    resumed_indices = np.array([int(k) for k in annotations_dict.keys()])
    resumed_labels = np.array(list(annotations_dict.values()))

    active_learner.initialize_data(resumed_indices, resumed_labels)

else:
    # Filter for articles that contain at least four of the above keywords.
    # This is a somewhat arbitrary number that was experimentally determined:
    # filtering on articles that contained at least one returned too many results to be useful,
    # while filtering on articles that contained all of them returned 0 results.
    # It doesn't really matter that much anyway, since this is only to get some useful seed labels.
    # The real classification is done later, after labeling.

    print(
        "No existing progress found. Starting fresh initialization with keyword filtering..."
    )

    candidate_indices = []

    for idx, text in enumerate(raw_texts):
        print(
            f"Scanning item {idx+1}/{len(raw_texts)} for keyword matches...",
            end="\r",
        )

        matches = KEYWORDS_PATTERN.findall(text)
        match_count = len(matches)


        if match_count >= 1:
            candidate_indices.append(idx)

    print(
        f"Found {len(candidate_indices)} potential hits out of {len(raw_texts)} total texts using keyword search."
    )

    # Randomly select 25 samples from this concentrated list, falling back to random if not enough hits.
    # Here, too, the number 25 was experimentally decided; using only 5 seed articles turned out not to provide much of a boost.
    if len(candidate_indices) >= 25:
        initial_indices = np.random.choice(candidate_indices, size=25, replace=False)
    else:
        print("Not enough keyword hits! Falling back to random initialization.")
        initial_indices = random_initialization(train_dataset, n_samples=25)

    seed_labels = []

    for count, idx in enumerate(initial_indices):
        print(f"\nSeed Item {count+1}/25")
        print(preview_text(raw_texts[idx]))
        print("")

        label = input(
            "Label (0 for Not Relevant, 1 for Relevant (Dutch culinary culture)): "
        )
        if label not in ["0", "1"]:
            raise ValueError("Invalid label! Please enter 0 or 1.")
        seed_labels.append(int(label))

        print("\n\n\n\n\n")
        print_divider()

    active_learner.initialize_data(initial_indices, np.array(seed_labels))

    for idx, lbl in zip(initial_indices, seed_labels):
        annotations_dict[str(idx)] = int(lbl)

    with open(REDDIT_ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)


# ==========================================
# 4. The Active Learning loop
# ==========================================
samples_per_query = 50

print_header(f"Starting labeling session (Batch size: {samples_per_query})...\n")

# Step A: The learner calculates uncertainties and asks for the most informative texts
queried_indices = active_learner.query(num_samples=samples_per_query)

# Step B: manual labeling
current_labels = []
quit_requested = False

for count, idx in enumerate(queried_indices):
    # Displaying how many of the above-defined keywords are present in this article makes labeling easier.
    # For instance, trying to manually scan a 2000-word article about an athlete to find out if it contains an utterance about Dutch food is quite difficult.
    # But if we can see that it contains 0 out of the 16 keywords, it's immediately obvious that it's almost certainly not relevant, and we can label it as such without having to read the entire thing.
    matches = set(m.lower() for m in KEYWORDS_PATTERN.findall(raw_texts[idx]))
    keywords_count = len(matches)
    print(
        f"\nItem {count+1}/{samples_per_query}. {keywords_count}/{len(KEYWORDS)} keywords."
    )
    print(preview_text(raw_texts[idx]))
    print("")

    label = input("Label (0=No, 1=Yes, 'q'=Save and Quit): ")

    if label.lower() == "q":
        quit_requested = True
        break

    if label not in ["0", "1"]:
        raise ValueError("Invalid label! Please enter 0, 1, or q.")

    current_labels.append(int(label))

    print("\n\n\n\n\n\n\n\n")
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

    with open(REDDIT_ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)

    print_header("Early exit requested. Partial batch saved to JSON. Wrapping up...")

else:
    # Full batch completed. Update the model and save to JSON.
    active_learner.update(np.array(current_labels))

    for idx, lbl in zip(queried_indices, current_labels):
        annotations_dict[str(idx)] = int(lbl)

    with open(REDDIT_ANNOTATIONS_FILE, "w") as f:
        json.dump(annotations_dict, f)

print(f"Total labeled: {len(annotations_dict)}")

print("\nSaving the model...")

final_model = active_learner.classifier.model  # type: ignore

joblib.dump(final_model, REDDIT_MODEL_FILE)

print("Saved successfully!")
