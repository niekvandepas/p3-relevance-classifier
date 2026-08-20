import re
import json
import os
import random
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from project_types import DelpherItem
from util import preview_text, print_divider, print_header

load_dotenv()


def get_data_path() -> Path:
    """
    Returns the Path to the local Delpher .ndjson file, as configured in .env.
    Delpher data is copyrighted and cannot be stored on Hugging Face, so it must
    be available locally.
    """
    DELPHER_DATA_FILE = os.environ.get("DELPHER_DATA_FILE")

    if not DELPHER_DATA_FILE:
        raise ValueError(
            "DELPHER_DATA_FILE environment variable not set. Please set it in your .env file to the path of the local Delpher .ndjson file."
        )

    data_path = Path(DELPHER_DATA_FILE)

    if not data_path.exists():
        raise ValueError(
            f"DELPHER_DATA_FILE '{data_path}' does not exist. Please check the path in your .env file."
        )

    return data_path


def import_data(data_file: Path, limit: int | None = None) -> list[DelpherItem]:
    results = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
            if len(results) == limit:
                break
    return results


KEYWORDS_PATTERN = re.compile(
    r"\bnederlandse keuken\b"
    r"|"
    r"\bnederlandse eetcultuur\b"
    r"|"
    r"\bnederlands eten\b"
    r"|"
    r"\bnederlandse cuisine\b"
    r"|"
    r"\bnederlandse maaltijd\b"
    r"|"
    r"\bhollandse maaltijd\b"
    r"|"
    r"\bhollandse keuken\b"
    r"|"
    r"\bhollands eten\b"
    r"|"
    r"\bnederlandse kost\b"
    r"|"
    r"\bhollandse kost\b"
    r"|"
    r"\bvaderlandse keuken\b"
    r"|"
    r"\bvaderlansd eten\b"
    r"|"
    r"\bvaderlandse kost\b",
    flags=re.IGNORECASE,
)

DELPHER_LLM_ANNOTATIONS_FILE = "annotations/delpher_manual_eval_labels.json"

print("Loading Delpher data from local .ndjson file")
delpher_items_data_path = get_data_path()

delpher_items = import_data(delpher_items_data_path)


filtered_items = []

for item in tqdm(delpher_items):
    if KEYWORDS_PATTERN.search(item["plain_text"]) or KEYWORDS_PATTERN.search(
        item.get("title", "")
    ):
        filtered_items.append(item)

print(f"Filtered to {len(filtered_items)} items matching keywords")
LABELING_RANDOM_SEED = int(os.environ.get("LABELING_RANDOM_SEED", "42"))
random.Random(LABELING_RANDOM_SEED).shuffle(filtered_items)
print(f"Sampling items with random seed: {LABELING_RANDOM_SEED}")

BOLD = "\033[1m"
RESET = "\033[0m"

annotations_dict = {}

if Path(DELPHER_LLM_ANNOTATIONS_FILE).exists():
    with open(DELPHER_LLM_ANNOTATIONS_FILE, "r") as f:
        annotations_dict = json.load(f)

already_labeled = sum(
    1 for item in filtered_items if item["identifier"] in annotations_dict
)
print_header(
    f"{already_labeled}/{len(filtered_items)} items already labeled. "
    f"Annotations file: {DELPHER_LLM_ANNOTATIONS_FILE}"
)

quit_requested = False

for index, delpher_item in enumerate(filtered_items, start=1):
    if delpher_item["identifier"] in annotations_dict:
        continue

    text = delpher_item["plain_text"]

    print_divider()
    print(
        f"Item {index}/{len(filtered_items)} (labeled so far: {len(annotations_dict)})"
    )
    print(f"ID: {delpher_item['identifier']}")
    print("")

    print(delpher_item["title"].upper())
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
    annotations_dict[delpher_item["identifier"]] = int(label)

    print("\n\n\n\n\n\n\n\n")
    print_divider()

with open(DELPHER_LLM_ANNOTATIONS_FILE, "w") as f:
    json.dump(annotations_dict, f)

if quit_requested:
    print_header("Early exit requested. Partial batch saved to JSON. Wrapping up...")

print(f"Total labeled: {len(annotations_dict)}")
