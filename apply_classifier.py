from collections.abc import Iterator
import json
import os
from pathlib import Path
import warnings

import dotenv
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer

from constants import REDDIT_LANGUAGE
from project_types import RedditItem

dotenv.load_dotenv()

HF_REPO_ID = os.environ.get("HF_REPO_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")

script_dir = Path(__file__).parent
results_dir = script_dir / "artifacts" / "results" / "llm"

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names*"
)  # Ignore LBGM warnings about feature names for clean output


def classify_dataset(data: Iterator[RedditItem]) -> None:
    #    config: Sentence-Bert embeddings for LightGBM model with {'learning_rate': 0.05, 'n_estimators': 200, 'num_leaves': 31}

    # --- Load unanimous LLM labels as training data ---
    qwen_df = pd.read_json(
        results_dir / "results-nl-Qwen-Qwen2.5-7B-Instruct-20260514-1537.ndjson",
        lines=True,
    )
    mistral_df = pd.read_json(
        results_dir
        / "results-nl-mistralai-Mistral-Medium-3.5-128B-20260514-1653.ndjson",
        lines=True,
    )
    df_full = pd.merge(
        qwen_df, mistral_df[["label", "id"]], on="id", suffixes=("_Qwen", "_Mistral")
    )
    df_train = df_full[df_full["label_Qwen"] == df_full["label_Mistral"]].copy()
    df_train = df_train.dropna(subset=["text"])

    train_texts: list[str] = df_train["text"].tolist()
    y_train: np.ndarray = df_train["label_Qwen"].astype(int).values  # type: ignore

    print(f"Training on {len(train_texts):,} unanimously-labeled LLM items...")

    # --- Encode with Sentence-BERT ---
    encoder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    x_train = encoder.encode(train_texts, show_progress_bar=True)

    # --- Fit LightGBM with the specified config ---
    clf = LGBMClassifier(
        learning_rate=0.05,
        n_estimators=200,
        num_leaves=31,
        class_weight="balanced",
        n_jobs=-1,
        verbosity=-1,
    )
    clf.fit(x_train, y_train)

    # --- Classify all items in batches and to files ---
    output_file_relevant = (
        script_dir
        / "artifacts"
        / "results"
        / "classification"
        / f"reddit-classified-{REDDIT_LANGUAGE}-relevant.ndjson"
    )
    output_file_irrelevant = (
        script_dir
        / "artifacts"
        / "results"
        / "classification"
        / f"reddit-classified-{REDDIT_LANGUAGE}-irrelevant.ndjson"
    )

    output_file_relevant.parent.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 1000
    total_processed = 0
    total_relevant = 0
    batch_items: list[RedditItem] = []

    print(f"Running inference, writing to {results_dir} ...")

    with open(output_file_relevant, "w", encoding="utf-8") as relevant_out_f:
        with open(output_file_irrelevant, "w", encoding="utf-8") as irrelevant_out_f:

            def flush_batch() -> int:
                nonlocal batch_items

                texts = [item["text"] for item in batch_items]

                try:
                    # Fast path
                    x_batch = encoder.encode(texts, show_progress_bar=False)
                    preds = clf.predict(x_batch)

                    items_and_preds = zip(batch_items, preds)

                # Sometimes sentence transformers fails: https://github.com/huggingface/sentence-transformers/issues/3758
                except ValueError as e:
                    print(f"Batch failed, retrying individually: {e}")

                    valid_items = []
                    valid_embeddings = []

                    for item in batch_items:
                        try:
                            embedding = encoder.encode(
                                [item["text"]],
                                show_progress_bar=False,
                            )[0]

                            valid_items.append(item)
                            valid_embeddings.append(embedding)

                        except ValueError as e:
                            print(f"Skipping item {item['id']}: {e}")

                    if not valid_items:
                        batch_items = []
                        return 0

                    preds = clf.predict(np.array(valid_embeddings))

                    items_and_preds = zip(valid_items, preds)

                count = 0

                for item, pred in items_and_preds:
                    if pred == 1:
                        relevant_out_f.write(
                            json.dumps(
                                {"id": item["id"], "text": item["text"]}
                            ) + "\n"
                        )
                        count += 1
                    else:
                        irrelevant_out_f.write(
                            json.dumps(
                                {"id": item["id"], "text": item["text"]}
                            ) + "\n"
                        )

                batch_items = []

                return count

            for item in data:
                if not item.get("text"):
                    continue
                batch_items.append(item)
                total_processed += 1
                if len(batch_items) >= BATCH_SIZE:
                    total_relevant += flush_batch()
                    if total_processed % 50_000 == 0:
                        print(f"Processed {total_processed:,} items...")

            if batch_items:
                total_relevant += flush_batch()

    print(
        f"\nDone: {total_processed:,} items processed, "
        f"{total_relevant:,} relevant items written to {output_file_relevant}"
    )


def get_data_path(file_type: str, language: str) -> Path:
    """
    Returns a Path to the local cached version of the HuggingFace file.
    file_type: 'posts' or 'comments'
    language: 'en' or 'nl'
    """
    filename = f"reddit-{language}-{file_type}.ndjson"

    # This will download the file if missing, or return the path if it exists
    cached_path: str = hf_hub_download(
        repo_id=HF_REPO_ID, filename=filename, repo_type="dataset", token=HF_TOKEN  # type: ignore
    )

    return Path(cached_path)


def iter_reddit_files():
    for file_type in ["posts", "comments"]:
        yield get_data_path(file_type, REDDIT_LANGUAGE)


def iter_reddit_items() -> Iterator[RedditItem]:
    for path in iter_reddit_files():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


if __name__ == "__main__":
    data = iter_reddit_items()

    classify_dataset(data)
