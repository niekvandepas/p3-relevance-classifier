import json
import warnings
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import pandas as pd
from lightgbm import LGBMClassifier
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from huggingface_hub import hf_hub_download
from constants import REDDIT_ANNOTATIONS_FILE, REDDIT_LANGUAGE
from dataclasses import dataclass
import dotenv

dotenv.load_dotenv()


HF_REPO_ID = os.environ.get("HF_REPO_ID")

if not HF_REPO_ID:
    raise ValueError(
        "HF_REPO_ID environment variable is not set. Please set it to your HuggingFace repository ID."
    )

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN environment variable is not set. Please set it to your HuggingFace access token."
    )


@dataclass
class EvaluationResult:
    model_name: str
    cv_f1: float
    test_f1_relevant: float
    test_f1_irrelevant: float
    cohen_kappa: float
    best_params: str


@dataclass
class BranchResults:
    tfidf: list[EvaluationResult]
    sentence_bert: list[EvaluationResult]
    robbert: list[EvaluationResult]


warnings.filterwarnings(
    "ignore", message="X does not have valid feature names*"
)  # Ignore LBGM warnings about feature names for clean output

script_dir = Path(__file__).parent
results_dir = script_dir / "artifacts" / "results" / "llm"
gold_annotations_file = script_dir / "annotations" / "manual_eval_labels.json"


def evaluate_human_labeled_ml_models(
    id_to_text_mapping: dict[str, str],
) -> BranchResults:
    print("=====> Evaluating ML models trained on human-labeled data\n")

    al_labels_series = pd.read_json(
        REDDIT_ANNOTATIONS_FILE, orient="index", typ="series"
    )
    active_learning_labeled_df = al_labels_series.reset_index().rename(
        columns={"index": "id", 0: "label"}
    )
    active_learning_labeled_df["text"] = active_learning_labeled_df["id"].map(
        id_to_text_mapping
    )

    gold_df = (
        pd.read_json(gold_annotations_file, orient="index", typ="series")
        .reset_index()
        .rename(columns={"index": "id", 0: "gold_label"})
    )

    gold_df["text"] = gold_df["id"].map(id_to_text_mapping)

    df_train = active_learning_labeled_df[
        ~active_learning_labeled_df["id"].isin(gold_df["id"])
    ].copy()

    df_train["final_label"] = df_train["label"].astype(int)

    df_train = df_train.dropna(subset=["text"])
    df_test = gold_df.dropna(subset=["text"])

    train_texts: list[str] = df_train["text"].tolist()
    test_texts: list[str] = df_test["text"].tolist()

    y_train: np.ndarray = df_train["final_label"].values  # type: ignore
    y_test: np.ndarray = df_test["gold_label"].values  # type: ignore

    tfidf_results = evaluate_tfidf(train_texts, test_texts, y_train, y_test)
    sentencebert_results = evaluate_sentencebert(
        train_texts, test_texts, y_train, y_test
    )
    robbert_results = evaluate_robbert(train_texts, test_texts, y_train, y_test)
    return BranchResults(
        tfidf=tfidf_results,
        sentence_bert=sentencebert_results,
        robbert=robbert_results,
    )


def evaluate_llm_labeled_ml_models() -> BranchResults:
    print("=====> Evaluating ML models trained on LLM-labeled data\n")
    qwen25_file_name = "results-nl-Qwen-Qwen2.5-7B-Instruct-20260514-1537.ndjson"
    mistral35_file_name = (
        "results-nl-mistralai-Mistral-Medium-3.5-128B-20260514-1653.ndjson"
    )

    qwen_df = pd.read_json(results_dir / qwen25_file_name, lines=True)
    mistral_df = pd.read_json(results_dir / mistral35_file_name, lines=True)

    df_full_llm = pd.merge(
        qwen_df, mistral_df[["label", "id"]], on="id", suffixes=("_Qwen", "_Mistral")
    )
    df_unanimous_llm = df_full_llm[
        df_full_llm["label_Qwen"] == df_full_llm["label_Mistral"]
    ].copy()

    gold_df = (
        pd.read_json(gold_annotations_file, orient="index", typ="series")
        .reset_index()
        .rename(columns={"index": "id"})
    )

    df_train_llm = df_unanimous_llm[~df_unanimous_llm["id"].isin(gold_df["id"])].copy()
    df_train_llm["final_label"] = df_train_llm["label_Qwen"].astype(int)

    df_test_llm = pd.merge(
        gold_df.rename(columns={0: "gold_label"}),
        df_full_llm[["id", "text"]],
        on="id",
        how="inner",
    )

    train_texts: list[str] = df_train_llm["text"].tolist()
    test_texts: list[str] = df_test_llm["text"].tolist()

    y_train: np.ndarray = df_train_llm["final_label"].values  # type: ignore
    y_test: np.ndarray = df_test_llm["gold_label"].values  # type: ignore

    tfidf_results = evaluate_tfidf(train_texts, test_texts, y_train, y_test)
    sentencebert_results = evaluate_sentencebert(
        train_texts, test_texts, y_train, y_test
    )
    robbert_results = evaluate_robbert(train_texts, test_texts, y_train, y_test)

    return BranchResults(
        tfidf=tfidf_results,
        sentence_bert=sentencebert_results,
        robbert=robbert_results,
    )


def evaluate_tfidf(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[EvaluationResult]:
    print("Encoding TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = tfidf_vectorizer.fit_transform(train_texts)
    x_test = tfidf_vectorizer.transform(test_texts)

    print(f"\nTraining set size: {x_train.shape[0]}")
    print(f"Testing set size: {x_test.shape[0]}")

    print("\nRunning Grid Search Cross-Validation...")

    # Define the models and the parameters you want to test
    model_configs = {
        "Naive Bayes": {
            "model": MultinomialNB(),
            "params": {"alpha": [0.1, 0.5, 1.0, 2.0]},
        },
        "SVM (Linear)": {
            "model": SVC(class_weight="balanced"),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear"]},
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {"C": [0.1, 1, 10]},
        },
    }

    results: list[EvaluationResult] = []
    confusion_matrices = {}

    # Loop through each model and test its parameters
    for model_name, config in model_configs.items():
        classifier = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        # Fit strictly on the training data
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)  # type: ignore

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        confusion_matrices[model_name] = matrix
        results.append(
            EvaluationResult(
                model_name=model_name,
                cv_f1=round(classifier.best_score_, 4),
                test_f1_relevant=round(report["1"]["f1-score"], 4),  # type: ignore
                test_f1_irrelevant=round(report["0"]["f1-score"], 4),  # type: ignore
                cohen_kappa=round(kappa, 4),
                best_params=str(classifier.best_params_),
            )
        )

    print("=====> Confusion Matrices (TFIDF)")

    for model_name, cm in confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"                       Predicted 0    Predicted 1")
        print(f"Actual 0 (Irrelevant)  {cm[0][0]:<14} {cm[0][1]}")
        print(f"Actual 1 (Relevant)    {cm[1][0]:<14} {cm[1][1]}")

    print("\n=====> Hyperparameter Tuning Results (TFIDF)")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="cv_f1", ascending=False)
    print(results_df.to_string(index=False))

    return results


def evaluate_sentencebert(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[EvaluationResult]:
    print("\n=======> Running sentence-bert model")

    from sentence_transformers import SentenceTransformer

    print("Encoding labeled items with Sentence-BERT...")
    bert_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    x_train_llm = bert_model.encode(train_texts, show_progress_bar=True)
    x_test_llm = bert_model.encode(test_texts, show_progress_bar=True)

    # 5. Update model_configs (MultinomialNB won't work with BERT vectors)
    sentence_bert_model_configs = {
        "SVM (Linear)": {
            "model": SVC(class_weight="balanced"),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear"]},
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {"C": [0.1, 1, 10]},
        },
        "LightGBM": {
            "model": LGBMClassifier(class_weight="balanced", n_jobs=-1, verbosity=-1),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 50],
            },
        },
    }

    sentence_bert_results: list[EvaluationResult] = []
    sentence_bert_confusion_matrices = {}

    # Loop through each model and test its parameters
    for model_name, config in sentence_bert_model_configs.items():
        classifier = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        # Fit strictly on the training data
        classifier.fit(x_train_llm, y_train)  # type: ignore

        y_pred = classifier.predict(x_test_llm)

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        sentence_bert_confusion_matrices[model_name] = matrix
        sentence_bert_results.append(
            EvaluationResult(
                model_name=model_name,
                cv_f1=round(classifier.best_score_, 4),
                test_f1_relevant=round(report["1"]["f1-score"], 4),  # type: ignore
                test_f1_irrelevant=round(report["0"]["f1-score"], 4),  # type: ignore
                cohen_kappa=round(kappa, 4),
                best_params=str(classifier.best_params_),
            )
        )

    print("=====> Confusion Matrices (Sentence-Bert)")

    for model_name, cm in sentence_bert_confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"                       Predicted 0    Predicted 1")
        print(f"Actual 0 (Irrelevant)  {cm[0][0]:<14} {cm[0][1]}")
        print(f"Actual 1 (Relevant)    {cm[1][0]:<14} {cm[1][1]}")

    print("\n=====> Hyperparameter Tuning Results (Sentence-Bert)")

    results_df = pd.DataFrame(sentence_bert_results)
    results_df = results_df.sort_values(by="cv_f1", ascending=False)
    print(results_df.to_string(index=False))

    return sentence_bert_results


def evaluate_robbert(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> list[EvaluationResult]:
    print("\n=======> Running RoBERTa model")

    from sentence_transformers import SentenceTransformer

    print("Encoding labeled items with RoBBERT...")
    robbert_model = SentenceTransformer(
        "NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers"
    )
    x_train_llm = robbert_model.encode(train_texts, show_progress_bar=True)
    x_test_llm = robbert_model.encode(test_texts, show_progress_bar=True)

    # 5. Update model_configs (MultinomialNB won't work with BERT vectors)
    robbert_model_configs = {
        "SVM (Linear)": {
            "model": SVC(class_weight="balanced"),
            "params": {"C": [0.1, 1, 10], "kernel": ["linear"]},
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "params": {"C": [0.1, 1, 10]},
        },
        "LightGBM": {
            "model": LGBMClassifier(class_weight="balanced", n_jobs=-1, verbosity=-1),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 50],
            },
        },
    }

    robbert_results = []
    robbert_confusion_matrices = {}

    # Loop through each model and test its parameters
    for model_name, config in robbert_model_configs.items():
        classifier = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        # Fit strictly on the training data
        classifier.fit(x_train_llm, y_train)  # type: ignore

        y_pred = classifier.predict(x_test_llm)

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        robbert_confusion_matrices[model_name] = matrix
        robbert_results.append(
            EvaluationResult(
                model_name=model_name,
                cv_f1=round(classifier.best_score_, 4),
                test_f1_relevant=round(report["1"]["f1-score"], 4),  # type: ignore
                test_f1_irrelevant=round(report["0"]["f1-score"], 4),  # type: ignore
                cohen_kappa=round(kappa, 4),
                best_params=str(classifier.best_params_),
            )
        )

    print("=====> Confusion Matrices (Robbert)")

    for model_name, cm in robbert_confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"                       Predicted 0    Predicted 1")
        print(f"Actual 0 (Irrelevant)  {cm[0][0]:<14} {cm[0][1]}")
        print(f"Actual 1 (Relevant)    {cm[1][0]:<14} {cm[1][1]}")

    print("\n=====> Hyperparameter Tuning Results (Robbert)")

    results_df = pd.DataFrame(robbert_results)
    results_df = results_df.sort_values(by="cv_f1", ascending=False)
    print(results_df.to_string(index=False))

    return robbert_results


def load_hf_text_mapping() -> dict[str, str]:
    """
    Downloads and reads the raw posts and comments files from the
    HuggingFace repo to create a definitive id -> text mapping.
    """
    print("Building master text mapping directly from HuggingFace dataset...")
    language = REDDIT_LANGUAGE

    master_lookup = {}

    for file_type in ["posts", "comments"]:
        filename = f"reddit-{language}-{file_type}.ndjson"
        cached_path = hf_hub_download(
            repo_id=HF_REPO_ID, filename=filename, repo_type="dataset", token=HF_TOKEN  # type: ignore
        )

        with open(cached_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("id") and item.get("text"):
                    master_lookup[item["id"]] = item["text"]

    print(f"Text mapping built successfully with {len(master_lookup):,} total rows.")
    return master_lookup


if __name__ == "__main__":
    hf_master_lookup = load_hf_text_mapping()

    llm_results = evaluate_llm_labeled_ml_models()
    human_results = evaluate_human_labeled_ml_models(hf_master_lookup)

    best_model = max(
        llm_results.tfidf
        + llm_results.sentence_bert
        + llm_results.robbert
        + human_results.tfidf
        + human_results.sentence_bert
        + human_results.robbert,
        key=lambda r: r.cohen_kappa,
    )

    print(best_model)
