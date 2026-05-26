from typing import TypedDict
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from project_types import RedditItem
from constants import REDDIT_LANGUAGE
from util import preview_text, print_divider, print_header


def get_data_path(language: str) -> Path:
    """
    Returns a Path to the local cached version of the HuggingFace file.
    file_type: 'posts' or 'comments'
    language: 'en' or 'nl'
    """
    filename = f"reddit-{language}-posts_and_comments-filtered_NLsekeuken-eten-culinair-sample_200.ndjson"

    # This will download the file if missing, or return the path if it exists
    cached_path = hf_hub_download(
        repo_id="niekvdpas/reddit-languages-data",
        filename=filename,
        repo_type="dataset",
    )

    return Path(cached_path)


def import_data(data_file: Path, limit: int | None = None) -> list[RedditItem]:
    results = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
            if len(results) == limit:
                break
    return results


REDDIT_LLM_ANNOTATIONS_FILE = "annotations/manual_eval_labels.json"

print("Fetching data from HuggingFace Hub (or cache if available)")
reddit_items_data_path = get_data_path(REDDIT_LANGUAGE)

reddit_items = import_data(reddit_items_data_path)

annotations_dict = {}

if Path(REDDIT_LLM_ANNOTATIONS_FILE).exists():
    with open(REDDIT_LLM_ANNOTATIONS_FILE, "r") as f:
        annotations_dict = json.load(f)

# Removed current_labels list entirely
quit_requested = False

for reddit_item in reddit_items:
    if reddit_item["id"] in annotations_dict:
        continue

    text = reddit_item["text"]

    print(preview_text(text))
    print("")

    while True:
        label = input("Label (0=No, 1=Yes, 'q'=Save and Quit): ").strip().lower()
        if label in ["0", "1", "q"]:
            break
        print("Invalid input. Please enter 0, 1, or q.")

    if label == "q":
        quit_requested = True
        break

    # Immediately store the label in the dictionary under the correct ID
    annotations_dict[reddit_item["id"]] = int(label)

    print("\n\n\n\n\n\n\n\n")
    print_divider()

with open(REDDIT_LLM_ANNOTATIONS_FILE, "w") as f:
    json.dump(annotations_dict, f)

if quit_requested:
    print_header("Early exit requested. Partial batch saved to JSON. Wrapping up...")

print(f"Total labeled: {len(annotations_dict)}")
