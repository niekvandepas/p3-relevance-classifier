from datetime import datetime
import json
from pathlib import Path
import random
import time
from typing import TypedDict

import httpx
from huggingface_hub import hf_hub_download
from ollama import chat
from ollama import ChatResponse
import ollama
from dotenv import load_dotenv
import os

from tqdm import tqdm

from constants import REDDIT_LANGUAGE


class RedditItem(TypedDict):
    id: str
    text: str


def import_data(data_file: Path, limit: int | None = None) -> list[RedditItem]:
    results = []

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))

            if len(results) == limit:
                break

    return results


def get_data_path(file_type: str, language: str) -> Path:
    """
    Returns a Path to the local cached version of the HuggingFace file.
    file_type: 'posts' or 'comments'
    language: 'en' or 'nl'
    """
    filename = f"reddit-{language}-{file_type}-sample_5000.ndjson"

    # This will download the file if missing, or return the path if it exists
    cached_path = hf_hub_download(
        repo_id="niekvdpas/reddit-languages-data",
        filename=filename,
        repo_type="dataset",
    )

    return Path(cached_path)


def build_prompt(reddit_text: str, language: str) -> list:
    if language not in ["en", "nl"]:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are 'en' and 'nl'."
        )

    if language == "nl":
        system_prompt = """Je bent een Nederlandse antropoloog die nationale identiteit en voedselcultuur bestudeert.

Je taak is om te bepalen of een tekst letterlijk over Nederlandse voedselcultuur of eetpraktijken gaat.

CRITERIA:

- RELEVANT (1): Vermeldingen van traditionele gerechten (stamppot, pannenkoeken), eetrituelen, voedselgebruiken of de sociale rol van eten in Nederland.

- IRRELEVANT (0): Voedselidiomen (bijv. "andere koek", "boter op het hoofd"), metaforen, boodschappenlijstjes of niet-gerelateerde onderwerpen.

Antwoord STRIKT met één cijfer: 1 of 0. Geen uitleg."""

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "De Nederlandse keuken (veel aardappelen, groenten etc) is niet bijzonder, maar wel beter dan de Amerikaanse. De cuisines van deze twee landen staan ergens onderaan de lijst.",
            },
            {"role": "assistant", "content": "1"},
            {
                "role": "user",
                "content": "De nieuwe wetgeving is echt andere koek, de overheid heeft boter op haar hoofd.",
            },
            {"role": "assistant", "content": "0"},
            {
                "role": "user",
                "content": "Ik heb net vier kaassoufflés bij de Febo gehaald, was 10 euro.",
            },
            {"role": "assistant", "content": "0"},
            {
                "role": "user",
                "content": "Ja, stamppot, maar of dat nou echt lekker te noemen is...",
            },
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": f"{reddit_text}"},
        ]

    elif language == "en":
        system_prompt = """You are a Dutch Anthropologist studying national identity and food culture.
Your task is to classify whether a text discusses literal Dutch culinary culture or food practices as they relate to identity.

CRITERIA:
- RELEVANT (1): Mentions of traditional dishes (stamppot, pannenkoeken), food rituals, dining habits, or the social role of food in the Netherlands.
- IRRELEVANT (0): Food-based idioms (e.g., "andere koek", "boter op het hoofd"), metaphors, transactional grocery lists, or completely unrelated topics.

Respond STRICTLY with a single digit: 1 or 0. Do not explain your reasoning."""

        return [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": "Dutch cuisine is really nothing special in my opinion.",
            },
            {"role": "assistant", "content": "1"},
            {
                "role": "user",
                "content": "They really had egg on their face when that happpened.",
            },
            {"role": "assistant", "content": "0"},
            {
                "role": "user",
                "content": "I just went out to dinner and paid €70... food prices are getting out of hand.",
            },
            {"role": "assistant", "content": "0"},
            {
                "role": "user",
                "content": "Sure, stamppot, but does anybody actually like that?",
            },
            {"role": "assistant", "content": "1"},
            {"role": "user", "content": f"{reddit_text}"},
        ]


def main():
    load_dotenv()

    if not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN environment variable not set. Please set it in your .env file with a HuggingFace API token."
        )

    # =========== Ollama setup ===========
    LLM_NAME = os.environ.get("LLM_NAME")

    if not LLM_NAME:
        raise ValueError(
            "LLM_NAME environment variable not set. Please set it in your .env file."
        )

    print("Waiting for Ollama server to be ready...")
    server_ready = False
    for _ in range(15):
        try:
            httpx.get("http://127.0.0.1:11434")
            server_ready = True
            break
        except httpx.ConnectError:
            time.sleep(1)

    if not server_ready:
        raise ConnectionError("Ollama server is not running. Did 'ollama serve' start?")

    print(
        f"Ensuring model '{LLM_NAME}' is downloaded (this may take a few minutes the first time)..."
    )
    ollama.pull(LLM_NAME)
    print("Model is ready!")

    # ========== Data import ===========

    LLM_REDDIT_DATA_SOURCE = os.environ.get("LLM_REDDIT_DATA_SOURCE")

    if LLM_REDDIT_DATA_SOURCE is None:
        raise ValueError(
            "LLM_REDDIT_DATA_SOURCE environment variable not set. Please set it to 'huggingface' or 'local' in your .env file."
        )

    if LLM_REDDIT_DATA_SOURCE == "huggingface":
        print("Fetching data from HuggingFace Hub (or cache if available)")
        reddit_posts_data_path = get_data_path("posts", REDDIT_LANGUAGE)
        reddit_comments_data_path = get_data_path("comments", REDDIT_LANGUAGE)
    elif LLM_REDDIT_DATA_SOURCE == "local":
        REDDIT_DATA_FOLDER = os.getenv("REDDIT_DATA_FOLDER")

        if not REDDIT_DATA_FOLDER:
            raise ValueError(
                "Please set the REDDIT_DATA_FOLDER environment variable in your .env file."
            )
        reddit_posts_data_filename = (
            f"reddit-{REDDIT_LANGUAGE}-posts-sample_5000.ndjson"
        )
        reddit_posts_data_path = Path(REDDIT_DATA_FOLDER) / reddit_posts_data_filename
        reddit_comments_data_filename = (
            f"reddit-{REDDIT_LANGUAGE}-comments-sample_5000.ndjson"
        )
        reddit_comments_data_path = (
            Path(REDDIT_DATA_FOLDER) / reddit_comments_data_filename
        )
    else:
        raise ValueError(
            f"Unsupported LLM_REDDIT_DATA_SOURCE: {LLM_REDDIT_DATA_SOURCE}. Supported values are 'huggingface' and 'local'."
        )

    # Import equal number of posts and comments
    posts = import_data(reddit_posts_data_path)
    comments = import_data(reddit_comments_data_path)

    all_items = posts + comments

    safe_model_name = LLM_NAME.replace("/", "-").replace(":", "-")

    results_file = (
        Path("artifacts")
        / "results"
        / f"results-{REDDIT_LANGUAGE}-{safe_model_name}-{datetime.now().strftime('%Y%m%d-%H%M')}.ndjson"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting classification with {LLM_NAME}")

    with open(results_file, "w", encoding="utf-8", buffering=1) as f_out:
        for item in tqdm(all_items):
            messages = build_prompt(item["text"], REDDIT_LANGUAGE)

            response: ChatResponse = chat(
                model=LLM_NAME,
                messages=messages,
                # Force deterministic output
                options={"temperature": 0},
            )

            classification = response.message.content.strip()  # type: ignore

            output_data = {
                "id": item["id"],
                "model": LLM_NAME,
                "label": classification,
                "text": item["text"],
            }

            f_out.write(json.dumps(output_data) + "\n")

    print(f"Analysis complete. Results saved to {results_file}")


if __name__ == "__main__":
    main()
