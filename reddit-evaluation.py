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

warnings.filterwarnings(
    "ignore", message="X does not have valid feature names*"
)  # Ignore LBGM warnings about feature names for clean output

script_dir = Path(__file__).parent


def evaluate_llm_labeled():
    results_dir = script_dir / "artifacts" / "results" / "llm"

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

    gold_annotations_file = script_dir / "annotations" / "manual_eval_labels.json"
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

    train_texts_llm: list[str] = df_train_llm["text"].tolist()
    test_texts_llm: list[str] = df_test_llm["text"].tolist()

    y_train_llm: np.ndarray = df_train_llm["final_label"].values  # type: ignore
    y_test_llm: np.ndarray = df_test_llm["gold_label"].values  # type: ignore

    evaluate_tfidf(train_texts_llm, test_texts_llm, y_train_llm, y_test_llm)
    evaluate_sentencebert(train_texts_llm, test_texts_llm, y_train_llm, y_test_llm)
    evaluate_robbert(train_texts_llm, test_texts_llm, y_train_llm, y_test_llm)


# def evaluate_researcher_labeled():
#     df = pd.read_json(REDDIT_ANNOTATIONS_FILE)

#     mistral_df = pd.read_json(results_dir / mistral35_file_name, lines=True)

#     df_full_llm = pd.merge(
#         qwen_df, mistral_df[["label", "id"]], on="id", suffixes=("_Qwen", "_Mistral")
#     )
#     df_unanimous_llm = df_full_llm[
#         df_full_llm["label_Qwen"] == df_full_llm["label_Mistral"]
#     ].copy()

#     gold_annotations_file = script_dir / "annotations" / "manual_eval_labels.json"
#     gold_df = (
#         pd.read_json(gold_annotations_file, orient="index", typ="series")
#         .reset_index()
#         .rename(columns={"index": "id"})
#     )

#     df_train_llm = df_unanimous_llm[~df_unanimous_llm["id"].isin(gold_df["id"])].copy()
#     df_train_llm["final_label"] = df_train_llm["label_Qwen"].astype(int)

#     df_test_llm = pd.merge(
#         gold_df.rename(columns={0: "gold_label"}),
#         df_full_llm[["id", "text"]],
#         on="id",
#         how="inner",
#     )

#     train_texts_llm: list[str] = df_train_llm["text"].tolist()
#     test_texts_llm: list[str] = df_test_llm["text"].tolist()

#     y_train_llm: np.ndarray = df_train_llm["final_label"].values  # type: ignore
#     y_test_llm: np.ndarray = df_test_llm["gold_label"].values  # type: ignore

#     evaluate_tfidf(train_texts_llm, test_texts_llm, y_train_llm, y_test_llm)
#     evaluate_sentencebert(train_texts_llm, test_texts_llm, y_train_llm, y_test_llm)
#     evaluate_robbert(train_texts_llm, test_texts_llm, y_train_llm, y_test_llm)


def evaluate_tfidf(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
):
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

    results = []
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

        y_pred = classifier.predict(x_test)

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        confusion_matrices[model_name] = matrix
        results.append(
            {
                "Model": model_name,
                "CV F1 Score": round(classifier.best_score_, 4),
                "Test F1 (Relevant)": round(report["1"]["f1-score"], 4),  # type: ignore
                "Test F1 (Not Relevant)": round(report["0"]["f1-score"], 4),  # type: ignore
                "Cohen's Kappa": round(kappa, 4),
                "Best Params": str(classifier.best_params_),
            }
        )

    print("=====> Confusion Matrices (TFIDF)")

    for model_name, cm in confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"                       Predicted 0    Predicted 1")
        print(f"Actual 0 (Irrelevant)  {cm[0][0]:<14} {cm[0][1]}")
        print(f"Actual 1 (Relevant)    {cm[1][0]:<14} {cm[1][1]}")

    print("\n=====> Hyperparameter Tuning Results (TFIDF)")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="CV F1 Score", ascending=False)
    print(results_df.to_string(index=False))


def evaluate_sentencebert(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
):
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

    sentence_bert_results = []
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
        classifier.fit(x_train_llm, y_train)

        y_pred = classifier.predict(x_test_llm)

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        sentence_bert_confusion_matrices[model_name] = matrix
        sentence_bert_results.append(
            {
                "Model": model_name,
                "CV F1 Score": round(classifier.best_score_, 4),
                "Test F1 (Relevant)": round(report["1"]["f1-score"], 4),  # type: ignore
                "Test F1 (Not Relevant)": round(report["0"]["f1-score"], 4),  # type: ignore
                "Cohen's Kappa": round(kappa, 4),
                "Best Params": str(classifier.best_params_),
            }
        )

    print("=====> Confusion Matrices (Sentence-Bert)")

    for model_name, cm in sentence_bert_confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"                       Predicted 0    Predicted 1")
        print(f"Actual 0 (Irrelevant)  {cm[0][0]:<14} {cm[0][1]}")
        print(f"Actual 1 (Relevant)    {cm[1][0]:<14} {cm[1][1]}")

    print("\n=====> Hyperparameter Tuning Results (Sentence-Bert)")

    results_df = pd.DataFrame(sentence_bert_results)
    results_df = results_df.sort_values(by="CV F1 Score", ascending=False)
    print(results_df.to_string(index=False))


def evaluate_robbert(
    train_texts: list[str],
    test_texts: list[str],
    y_train: np.ndarray,
    y_test: np.ndarray,
):
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
        classifier.fit(x_train_llm, y_train)

        y_pred = classifier.predict(x_test_llm)

        report = classification_report(y_test, y_pred, output_dict=True)
        matrix = confusion_matrix(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        robbert_confusion_matrices[model_name] = matrix
        robbert_results.append(
            {
                "Model": model_name,
                "CV F1 Score": round(classifier.best_score_, 4),
                "Test F1 (Relevant)": round(report["1"]["f1-score"], 4),  # type: ignore
                "Test F1 (Not Relevant)": round(report["0"]["f1-score"], 4),  # type: ignore
                "Cohen's Kappa": round(kappa, 4),
                "Best Params": str(classifier.best_params_),
            }
        )

    print("=====> Confusion Matrices (Robbert)")

    for model_name, cm in robbert_confusion_matrices.items():
        print(f"\n{model_name}:")
        print(f"                       Predicted 0    Predicted 1")
        print(f"Actual 0 (Irrelevant)  {cm[0][0]:<14} {cm[0][1]}")
        print(f"Actual 1 (Relevant)    {cm[1][0]:<14} {cm[1][1]}")

    print("\n=====> Hyperparameter Tuning Results (Robbert)")

    results_df = pd.DataFrame(robbert_results)
    results_df = results_df.sort_values(by="CV F1 Score", ascending=False)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    evaluate_llm_labeled()
    # evaluate_researcher_labeled()
