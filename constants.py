import os
from pathlib import Path
from dotenv import load_dotenv
from nltk.corpus import stopwords

load_dotenv()


# Delpher constants

DELPHER_MODEL_FILE_NAME = f"delpher_relevance_model.pkl"
DELPHER_MODEL_FILE = Path("artifacts/models") / DELPHER_MODEL_FILE_NAME

DELPHER_VECTORIZER_FILE_NAME = f"delpher_relevance_vectorizer.pkl"
DELPHER_VECTORIZER_FILE = Path("artifacts/models") / DELPHER_VECTORIZER_FILE_NAME

DELPHER_CACHE_FILE_NAME = f"delpher_dataset_cache.joblib"
DELPHER_CACHE_FILE = Path("artifacts/cache") / DELPHER_CACHE_FILE_NAME

DELPHER_ANNOTATIONS_FILE_NAME = f"delpher_annotations_progress.json"
DELPHER_ANNOTATIONS_FILE = Path("annotations") / DELPHER_ANNOTATIONS_FILE_NAME

# Reddit constants

REDDIT_LANGUAGE = "nl"

REDDIT_MODEL_FILE_NAME = f"reddit_relevance_model.{REDDIT_LANGUAGE}.pkl"
REDDIT_MODEL_FILE = Path("artifacts/models") / REDDIT_MODEL_FILE_NAME

REDDIT_VECTORIZER_FILE_NAME = f"reddit_relevance_vectorizer.{REDDIT_LANGUAGE}.pkl"
REDDIT_VECTORIZER_FILE = Path("artifacts/models") / REDDIT_VECTORIZER_FILE_NAME

REDDIT_CACHE_FILE_NAME = f"reddit_dataset_cache.{REDDIT_LANGUAGE}.joblib"
REDDIT_CACHE_FILE = Path("artifacts/cache") / REDDIT_CACHE_FILE_NAME

REDDIT_ANNOTATIONS_FILE_NAME = f"reddit_annotations_progress.{REDDIT_LANGUAGE}.json"
REDDIT_ANNOTATIONS_FILE = Path("annotations") / REDDIT_ANNOTATIONS_FILE_NAME


REDDIT_STOP_WORDS = (
    list(set(stopwords.words("dutch")).union({"mijn", "ik", "zijn", "was", "we"}))
    if REDDIT_LANGUAGE == "nl"
    else list(set(stopwords.words("english")))
)
