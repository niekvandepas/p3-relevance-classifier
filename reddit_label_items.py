import re
import random
import os
from typing import TypedDict
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
from tqdm import tqdm
from project_types import RedditItem
from constants import REDDIT_LANGUAGE
from util import preview_text, print_divider, print_header


def get_data_path(language: str) -> Path:
    """
    Returns a Path to the local cached version of the HuggingFace file.
    file_type: 'posts' or 'comments'
    language: 'en' or 'nl'
    """
    filename = f"reddit-{language}-keyword-filtered.ndjson"

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


REDDIT_LLM_ANNOTATIONS_FILE = (
    f"annotations/reddit_{REDDIT_LANGUAGE}_manual_eval_labels.json"
)

print("Fetching data from HuggingFace Hub (or cache if available)")
reddit_items_data_path = get_data_path(REDDIT_LANGUAGE)

reddit_items = import_data(reddit_items_data_path)


NL_PATTERN = re.compile(
    r"\b("
    r"eten|voedsel|koken|culinair|cuisine|gastronomisch|erfgoed|eetcultuur"
    r"|boerenkool|stamppot|hutspot|zuurkool|snert|erwtensoep"
    r"|oliebol|oliebollen|bitterbal|bitterballen|kroket|kroketten|frikandel|frikandellen"
    r"|stroopwafel|stroopwafels|hagelslag|drop|speculaas|ontbijtkoek|poffertjes"
    r"|haring|maatjesharing|kabeljauw|lekkerbek|lekkerbekje"
    r"|kaas|gouda|leidse|alkmaarse"
    r")\b"
    r"|"
    r"\b(nederlandse|hollandse|nederlands)\b.*\bkeuken\b"
    r"|"
    r"\bkeuken\b.*\b(nederlandse|hollandse|nederlands)\b",
    flags=re.IGNORECASE,
)

EN_PATTERN = re.compile(
    r"\b("
    r"food|cooking|cook|culinary|cuisine|gastronom|gastronomy|heritage|food\s*culture"
    r"|kale|boerenkool|stamppot|hutspot|sauerkraut|zuurkool|snert|erwtensoep|pea\s*soup"
    r"|oliebol|oliebollen|bitterbal|bitterballen|kroket|kroketten|croquette|croquettes|frikandel|frikandellen"
    r"|stroopwafel|stroopwafels|hagelslag|sprinkles|drop|licorice|speculaas|ontbijtkoek|poffertjes"
    r"|haring|herring|cod|kabeljauw|smelt|lekkerbek"
    r"|cheese|kaas|gouda|leidse|alkmaarse"
    r")\b"
    r"|"
    r"\b(dutch)\b.*\b(cuisine|cooking|food|kitchen)\b"
    r"|"
    r"\b(cuisine|cooking|food|kitchen)\b.*\b(dutch)\b",
    flags=re.IGNORECASE,
)

KEYWORDS_PATTERN = NL_PATTERN if REDDIT_LANGUAGE == "nl" else EN_PATTERN

BOLD = "\033[1m"
RESET = "\033[0m"
random.Random(42).shuffle(reddit_items)

annotations_dict = {}

if Path(REDDIT_LLM_ANNOTATIONS_FILE).exists():
    with open(REDDIT_LLM_ANNOTATIONS_FILE, "r") as f:
        annotations_dict = json.load(f)

quit_requested = False

for reddit_item in reddit_items:
    if reddit_item["id"] in annotations_dict:
        continue

    text = reddit_item["text"]

    preview = preview_text(text)
    highlighted_preview = KEYWORDS_PATTERN.sub(rf"{BOLD}\g<0>{RESET}", preview)
    print(highlighted_preview)
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
