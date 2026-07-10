print("Importing modules")
from datetime import datetime
import json
import math
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

# vllm does not build on macOS, so silence import error
from project_types import RedditItem
from vllm import LLM, SamplingParams  # type: ignore
from vllm.sampling_params import StructuredOutputsParams  # type: ignore

from transformers import AutoTokenizer

from dotenv import load_dotenv
import os

from tqdm import tqdm

from constants import REDDIT_LANGUAGE

load_dotenv()


def require_env(var_name: str, help_message: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise ValueError(help_message)
    return value


HF_REPO_ID = require_env(
    "HF_REPO_ID",
    "HF_REPO_ID environment variable not set. Please set it in your .env file.",
)
HF_TOKEN = require_env(
    "HF_TOKEN",
    "HF_TOKEN environment variable not set. Please set it in your .env file with a HuggingFace API token.",
)
HF_RESULTS_REPO_ID = os.environ.get(
    "HF_RESULTS_REPO_ID", "niekvdpas/p3-classification-results"
)


def upload_results_to_huggingface(results_file: Path, llm_name: str) -> None:
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=HF_RESULTS_REPO_ID, repo_type="dataset", exist_ok=True)

    path_in_repo = f"llm/{results_file.name}"

    api.upload_file(
        path_or_fileobj=str(results_file),
        path_in_repo=path_in_repo,
        repo_id=HF_RESULTS_REPO_ID,
        repo_type="dataset",
        commit_message=f"Add LLM classification results for {llm_name}",
    )

    print(
        f"Uploaded results file to Hugging Face dataset {HF_RESULTS_REPO_ID} at {path_in_repo}"
    )


LLM_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Short explanation of the choice, max 15 words.",
        },
        "label": {"type": "integer", "enum": [0, 1]},
    },
    "required": ["reasoning", "label"],
    "additionalProperties": False,
}


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
    filename = f"reddit-{language}-{file_type}-keyword-filtered.ndjson"

    # This will download the file if missing, or return the path if it exists
    cached_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        repo_type="dataset",
    )

    return Path(cached_path)


def build_prompt(reddit_text: str, language: str) -> list:
    if language == "nl":
        system_prompt = """Je bent een uiterst strikte Nederlandse antropoloog die de letterlijke, culinaire nationale identiteit bestudeert.

Je enige taak is om te bepalen of een tekst EXPLICIET over Nederlandse voedselcultuur, traditionele gerechten of eetgewoonten gaat. Je bent zeer streng: bij twijfel of indirecte links is het antwoord altijd 0.

CRITERIA VOOR RELEVANT (1):
- Specifieke Nederlandse gerechten of snacks (bijv. stamppot, bitterballen, frikandelbroodje, hagelslag).
- Kookgewoonten, receptdiscussies of eetcultuur in Nederlandse huishoudens.
- Discussies over wat "typisch Nederlands" eten is (bijv. patat vs. friet).

CRITERIA VOOR IRRELEVANT (0):
- Landbouw, voedselexport of boerderijen (bijv. "uienexport", "landbouwexport").
- Supermarktacties, logistiek of algemene voedselprijzen (bijv. "AH pannenzegels", "boodschappen doen").
- Horeca-industrie in het algemeen (bijv. "terrassen open", "kroegen dicht").
- Voedselidiomen of metaforen (bijv. "andere koek").
- Elke tekst die eten slechts terloops noemt zonder culturele context.

Antwoord uitsluitend in JSON-formaat. Geef eerst een korte 'reasoning' (maximaal 15 woorden) en daarna het 'label' (1 of 0)."""

        return [
            {"role": "system", "content": system_prompt},
            # Edge Case: Agriculture (0)
            {
                "role": "user",
                "content": "Nederlandse landbouwexport nooit eerder zo hoog",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Dit gaat over macro-economische export, niet over eetcultuur.", "label": 0}',
            },
            # Edge Case: Supermarket logic (0)
            {
                "role": "user",
                "content": "Iemand nog pannenzegels van de AH over om te delen?",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Dit is een supermarkt spaaractie, geen maaltijd of culinair gebruik.", "label": 0}',
            },
            # Clear Hit: Culinary Culture (1)
            {"role": "user", "content": "Broodje hagelslag met of zonder boter?"},
            {
                "role": "assistant",
                "content": '{"reasoning": "Discussie over de bereiding van een typisch Nederlands ontbijt.", "label": 1}',
            },
            # Edge Case: General hospitality/business (0)
            {
                "role": "user",
                "content": "Van terrasdirigent tot verwijderde lantaarnpalen: de horeca is er klaar voor",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Dit betreft horeca-logistiek en bedrijfsvoering, geen voedselcultuur.", "label": 0}',
            },
            # Clear Hit: Recipes/Cooking (1)
            {
                "role": "user",
                "content": "Recept voor bitterballen/kroketten? Ik ben opzoek naar iedereens favoriete recept...",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Vraag naar recepten voor traditionele Nederlandse snacks.", "label": 1}',
            },
            # The actual target
            {"role": "user", "content": f"{reddit_text}"},
        ]

    elif language == "en":
        # Note: You can apply the exact same logic for English if needed.
        # I have translated the strict criteria here.
        system_prompt = """You are an extremely strict Dutch Anthropologist studying literal culinary national identity.

Your only task is to determine if a text EXPLICITLY discusses Dutch food culture, traditional dishes, or dining habits. You are very strict: when in doubt or if the link is indirect, the answer is always 0.

CRITERIA FOR RELEVANT (1):
- Specific Dutch dishes or snacks (e.g., stamppot, bitterballen, frikandelbroodje, hagelslag).
- Cooking habits, recipe discussions, or food culture in Dutch households.
- Debates on "typical Dutch" food.

CRITERIA FOR IRRELEVANT (0):
- Agriculture, food exports, or farming.
- Supermarket logistics or general food prices.
- The hospitality industry in general (e.g., "terraces opening").
- Food idioms or metaphors.
- Any text mentioning food casually without cultural context.

Respond exclusively in JSON format. Provide a short 'reasoning' (maximum 15 words) first, followed by the 'label' (1 or 0)."""

        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": "Dutch agricultural exports have never been higher",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Discusses macroeconomic exports, not literal eating culture.", "label": 0}',
            },
            {
                "role": "user",
                "content": "Does anyone have spare Albert Heijn pan stamps to share?",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Supermarket loyalty program, unrelated to culinary habits.", "label": 0}',
            },
            {"role": "user", "content": "Broodje hagelslag with or without butter?"},
            {
                "role": "assistant",
                "content": '{"reasoning": "Debate about preparing a traditional Dutch breakfast item.", "label": 1}',
            },
            {
                "role": "user",
                "content": "From terrace managers to removed lampposts: the hospitality sector is ready",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Hospitality business logistics, not actual food culture.", "label": 0}',
            },
            {
                "role": "user",
                "content": "Recipe for bitterballen/kroketten? I am looking for everyone's favorite recipe...",
            },
            {
                "role": "assistant",
                "content": '{"reasoning": "Seeking a recipe for traditional Dutch snacks.", "label": 1}',
            },
            {"role": "user", "content": f"{reddit_text}"},
        ]
    else:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are 'en' and 'nl'."
        )


def main():
    print("In main() function")

    LLM_NAME = os.environ.get("LLM_NAME")

    if not LLM_NAME:
        raise ValueError(
            "LLM_NAME environment variable not set. Please set it in your .env file."
        )

    NUM_GPUS = os.environ.get("NUM_GPUS")
    if not NUM_GPUS:
        raise ValueError(
            "NUM_GPUS environment variable not set. Please set it in your .env file to the number of GPUs you want to use (e.g., 1, 2, 4)."
        )

    NUM_GPUS = int(NUM_GPUS)

    print(f"Using LLM: {LLM_NAME} with {NUM_GPUS} GPU(s)")

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

    posts = import_data(reddit_posts_data_path)
    comments = import_data(reddit_comments_data_path)

    all_items = posts + comments
    ratio_raw = os.environ.get("LLM_CLASSIFICATION_RATIO")
    if ratio_raw:
        try:
            ratio = float(ratio_raw)
        except ValueError as exc:
            raise ValueError(
                "Invalid LLM_CLASSIFICATION_RATIO. Use a number between 0 and 1, e.g. 0.1"
            ) from exc

        if not (0 < ratio <= 1):
            raise ValueError(
                "Invalid LLM_CLASSIFICATION_RATIO. Use a number between 0 and 1, e.g. 0.1"
            )

        count = max(1, math.ceil(len(all_items) * ratio))
        all_items = all_items[:count]
        print(
            f"Applying LLM_CLASSIFICATION_RATIO={ratio}; running on {len(all_items)} items."
        )
    else:
        print(f"LLM_CLASSIFICATION_RATIO not set; running full dataset ({len(all_items)} items).")

    safe_model_name = LLM_NAME.replace("/", "-").replace(":", "-")

    results_file = (
        Path("artifacts")
        / "results"
        / f"results-{REDDIT_LANGUAGE}-{safe_model_name}-{datetime.now().strftime('%Y%m%d-%H%M')}.ndjson"
    )
    results_file.parent.mkdir(parents=True, exist_ok=True)

    schema_str = json.dumps(LLM_JSON_SCHEMA)
    structured_params = StructuredOutputsParams(json=schema_str)

    # Configure generation parameters (temperature 0 for determinism, max_tokens 60 to allow reasoning)
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=60, structured_outputs=structured_params
    )

    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
    llm = LLM(model=LLM_NAME, tensor_parallel_size=NUM_GPUS)

    print(f"Applying chat templates to {len(all_items)} items...")
    formatted_prompts = []

    for item in tqdm(all_items, desc="Formatting"):
        if LLM_NAME == "BramVanroy/fietje-2-chat":
            item_text = item["text"][
                :2040
            ]  # Fietje has a 2048 token limit, so we truncate to be safe. We will recover the full text later in the analysis phase.
        else:
            item_text = item["text"]

        messages = build_prompt(item_text, REDDIT_LANGUAGE)
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    print(f"Starting batch classification with {LLM_NAME}...")

    # Pass the entire list at once. vLLM will handle internal continuous batching automatically.
    outputs = llm.generate(formatted_prompts, sampling_params)

    print("Writing results to disk...")
    with open(results_file, "w", encoding="utf-8", buffering=1) as f_out:
        for item, output in zip(all_items, outputs):
            raw_response = output.outputs[0].text.strip()

            # Safely parse the JSON output
            try:
                parsed_response = json.loads(raw_response)
                label = parsed_response.get("label", 0)
                reasoning = parsed_response.get("reasoning", "")
            except json.JSONDecodeError:
                # Fallback in case of a highly unusual parsing failure
                label = 0
                reasoning = "JSON parsing error"

            output_data = {
                "id": item["id"],
                "model": LLM_NAME,
                "label": label,
                "reasoning": reasoning,
                "text": item["text"],
            }

            f_out.write(json.dumps(output_data) + "\n")

    print(f"Analysis complete. Results saved to {results_file}")
    print("Uploading results to Hugging Face...")
    upload_results_to_huggingface(results_file, LLM_NAME)


if __name__ == "__main__":
    main()
