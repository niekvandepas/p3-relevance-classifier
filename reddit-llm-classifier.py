import json
from pathlib import Path
import random
from typing import TypedDict

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


def import_random_items_from(data_file: Path, limit: int) -> list[RedditItem]:
    with open(data_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    indices_to_pick = set(random.sample(range(total_lines), min(limit, total_lines)))

    results = []

    # Re-open the file because otherwise we'd have to seek back manually
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if i in indices_to_pick:
                results.append(json.loads(line))

            if len(results) == limit:
                break

    return results


# TODO update this prompt for the real analysis
def build_prompt(reddit_text: str, language: str) -> list:
    if language not in ["en", "nl"]:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are 'en' and 'nl'."
        )

    if language == "nl":
        system_prompt = """Je bent een Nederlandse antropoloog die nationale identiteit en voedselcultuur bestudeert.

Je taak is om te bepalen of een tekst letterlijk over Nederlandse voedselcultuur of eetpraktijken gaat in relatie tot identiteit.

CRITERIA:

- RELEVANT (1): Vermeldingen van traditionele gerechten (stamppot, pannenkoeken), eetrituelen, voedselgebruiken of de sociale rol van eten in Nederland.

- IRRELEVANT (0): Voedselidiomen (bijv. "andere koek", "boter op het hoofd"), metaforen, boodschappenlijstjes of niet-gerelateerde onderwerpen.

Antwoord STRIKT met één cijfer: 1 of 0. Geen uitleg."""

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
            "content": "Ik vind dat we op pakjesavond gewoon pepernoten en marsepein moeten eten, dat is traditie.",
        },
        {"role": "assistant", "content": "1"},
        {
            "role": "user",
            "content": "De nieuwe wetgeving is echt andere koek, de overheid heeft boter op haar hoofd.",
        },
        {"role": "assistant", "content": "0"},
        {"role": "user", "content": f"{reddit_text}"},
    ]


def main():
    load_dotenv()

    DATA_IMPORT_LIMIT = os.environ.get("LLM_ANALYSIS_DATA_IMPORT_LIMIT")

    if not DATA_IMPORT_LIMIT:
        raise ValueError(
            "LLM_ANALYSIS_DATA_IMPORT_LIMIT environment variable not set. Please set it in your .env file."
        )

    DATA_IMPORT_LIMIT = int(DATA_IMPORT_LIMIT)

    # =========== Ollama setup ===========
    LLM_NAME = os.environ.get("LLM_NAME")

    if not LLM_NAME:
        raise ValueError(
            "LLM_NAME environment variable not set. Please set it in your .env file."
        )

    print(
        f"Ensuring model '{LLM_NAME}' is downloaded (this may take a few minutes the first time)..."
    )
    ollama.pull(LLM_NAME)
    print("Model is ready!")

    # ========== Data import ===========
    REDDIT_DATA_FOLDER = os.getenv("REDDIT_DATA_FOLDER")

    if not REDDIT_DATA_FOLDER:
        raise ValueError(
            "Please set the REDDIT_DATA_FOLDER environment variable in your .env file."
        )

    reddit_posts_data_filename = f"reddit-{REDDIT_LANGUAGE}-posts.ndjson"
    reddit_posts_data_path = Path(REDDIT_DATA_FOLDER) / reddit_posts_data_filename
    reddit_comments_data_filename = f"reddit-{REDDIT_LANGUAGE}-comments.ndjson"
    reddit_comments_data_path = Path(REDDIT_DATA_FOLDER) / reddit_comments_data_filename

    # Import equal number of posts and comments
    posts = import_random_items_from(reddit_posts_data_path, int(DATA_IMPORT_LIMIT / 2))
    comments = import_random_items_from(
        reddit_comments_data_path, int(DATA_IMPORT_LIMIT / 2)
    )

    all_items = posts + comments

    for item in tqdm(all_items):
        messages = build_prompt(item["text"], REDDIT_LANGUAGE)

        response: ChatResponse = chat(
            model=LLM_NAME,
            messages=messages,
            # Force deterministic output
            options={"temperature": 0},
        )

        classification = response.message.content.strip()  # type: ignore
        tqdm.write(f"Text: {item['text'][:50]}... | Result: {classification}")


    print(response["message"]["content"])
    # or access fields directly from the response object
    print(response.message.content)


if __name__ == "__main__":
    main()
