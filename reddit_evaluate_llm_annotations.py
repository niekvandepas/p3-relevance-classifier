import pandas as pd
from pathlib import Path
import json

from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

script_dir = Path(__file__).parent
results_dir = script_dir / "artifacts" / "results" / "llm"

all_data = []

# Load all ndjson files
for file_path in results_dir.glob("*.ndjson"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

df = pd.DataFrame(all_data)

# Clean up: Force labels to strings and strip whitespace to ensure "1" == " 1"
df["label"] = df["label"].astype(str).str.strip()

# Pivot the data: This resolves the 20 duplicates by taking the first appearance
df_labels = df.pivot_table(
    index="id", columns="model", values="label", aggfunc="first"
).reset_index()

# Remove the name of the columns index (makes it look like a normal DataFrame)
df_labels.columns.name = None

# Grab the longest version of the text for each ID (recovers Fietje's truncations)
df["text_len"] = df["text"].str.len()
df_text = df.sort_values("text_len", ascending=False).drop_duplicates("id")[
    ["id", "text"]
]

# Final Merge
df_final = pd.merge(df_text, df_labels, on="id", how="left")

# Print the result
print(f"Merged {len(df_final)} unique Reddit items.")
print(f"Columns found: {df_final.columns.tolist()}")
print("\nFirst 5 rows:")
print(df_final.head())


model_cols = [c for c in df_final.columns if c not in ["id", "text"]]
agreement_series = (df_final[model_cols] == "1").sum(axis=1)
relevant_consensus = (agreement_series > 3).sum()
irrelevant_consensus = (agreement_series < 2).sum()

print(f"Items with >3 models agreeing on RELEVANT: {relevant_consensus}")
print(f"Items with >3 models agreeing on IRRELEVANT: {irrelevant_consensus}")
print(f"Total High-Confidence Items: {relevant_consensus + irrelevant_consensus}")


# 1. Calculate the number of '1' votes per row
model_cols = [c for c in df_final.columns if c not in ["id", "text"]]
agreement_series = (df_final[model_cols] == "1").sum(axis=1)

# 2. In a 5-model setup, 'No Agreement' (maximum conflict) is a 3-2 split.
# This means the sum of '1s' is either 2 or 3.
no_agreement_mask = (agreement_series == 2) | (agreement_series == 3)
no_agreement_count = no_agreement_mask.sum()

print(f"Total rows with no strong agreement (3-2 split): {no_agreement_count}")
print(
    f"Percentage of dataset in conflict: {(no_agreement_count / len(df_final)) * 100:.2f}%"
)

# 3. Peek at the 'messy' data to see why they disagree
print("\nExamples of high-conflict rows:")
print(df_final[no_agreement_mask][["text"] + model_cols].head())

# df_final.to_csv("master_llm_consensus.csv", index=False)

# ==========================================
# DEFINE PRECISION-FIRST ENSEMBLES
# ==========================================

# 1. Top 2 Unanimous (Two best models must both say 1)
top_2_models = ["Qwen/Qwen2.5-7B-Instruct", "mistralai/Mistral-Medium-3.5-128B"]
df_top2 = df_final[top_2_models].apply(pd.to_numeric, errors="coerce").fillna(0)
df_final["Ensemble_Top2_Unanimous"] = (df_top2.sum(axis=1) == 2).astype(int).astype(str)

# 2. Top 3 Unanimous (Adds Dutch specialist to the strict requirement)
top_3_models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-Medium-3.5-128B",
    "BramVanroy/GEITje-7B-ultra",
]
df_top3 = df_final[top_3_models].apply(pd.to_numeric, errors="coerce").fillna(0)
df_final["Ensemble_Top3_Unanimous"] = (df_top3.sum(axis=1) == 3).astype(int).astype(str)

# 3. Global Triad Unanimous (Strictest baseline across global architectures)
global_models = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-Medium-3.5-128B",
    "Qwen/Qwen3.6-27B",
]
df_global = df_final[global_models].apply(pd.to_numeric, errors="coerce").fillna(0)
df_final["Ensemble_Global_Triad_Unanimous"] = (
    (df_global.sum(axis=1) == 3).astype(int).astype(str)
)

# Add ensembles to the evaluation pipeline
precision_ensembles = [
    "Ensemble_Top2_Unanimous",
    "Ensemble_Top3_Unanimous",
    "Ensemble_Global_Triad_Unanimous",
]
model_cols.extend(precision_ensembles)


ANNOTATIONS_FILE = script_dir / "annotations" / "manual_eval_labels.json"

if ANNOTATIONS_FILE.exists():
    with open(ANNOTATIONS_FILE, "r") as f:
        gold_dict = json.load(f)

    # Convert dictionary to DataFrame
    df_gold = pd.DataFrame(list(gold_dict.items()), columns=["id", "gold_label"])
    df_gold["gold_label"] = df_gold["gold_label"].astype(int)

    # Inner merge to evaluate ONLY the items I manually labeled
    df_eval = pd.merge(df_final, df_gold, on="id", how="inner")

    print(f"\n" + "=" * 60)
    print(f" SKLEARN GOLD STANDARD EVALUATION ({len(df_eval)} items) ")
    print("=" * 60)

    eval_results = []

    for model in model_cols:
        print(f"\nDetailed Report for Model: {model}")

        # Parse model predictions safely to integers
        y_pred = pd.to_numeric(df_eval[model], errors="coerce").fillna(-1).astype(int)
        y_true = df_eval["gold_label"]

        # 1. Classification Report (Precision, Recall, F1 per class)
        print(classification_report(y_true, y_pred, labels=[0, 1], digits=3))

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        # 3. Cohen's Kappa (Inter-rater agreement between me and the model)
        kappa = cohen_kappa_score(y_true, y_pred)

        # Calculate high level accuracy metric
        accuracy = (tp + tn) / len(df_eval) if len(df_eval) > 0 else 0

        # Calculate summary precision/recall/f1 for class 1 to build summary dataframe
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_val = (
            2 * (precision_val * recall_val) / (precision_val + recall_val)
            if (precision_val + recall_val) > 0
            else 0
        )

        eval_results.append(
            {
                "Model": model,
                "Accuracy": round(accuracy, 3),
                "Precision (1)": round(precision_val, 3),
                "Recall (1)": round(recall_val, 3),
                "F1-Score (1)": round(f1_val, 3),
                "Cohen's Kappa": round(kappa, 3),
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
            }
        )

    # Summary table across all models
    df_metrics = pd.DataFrame(eval_results).set_index("Model")
    df_metrics.sort_values(by="F1-Score (1)", ascending=False, inplace=True)

    print("\n" + "=" * 60)
    print(" SUMMARY LEADERBOARD (Sorted by F1-Score for Relevant class) ")
    print("=" * 60)
    print(
        df_metrics[
            ["Accuracy", "Precision (1)", "Recall (1)", "F1-Score (1)", "Cohen's Kappa"]
        ].to_string()
    )

    # Save summary metrics to disk
    metrics_out = script_dir / "artifacts" / "llm_sklearn_evaluation.csv"
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(metrics_out)
    print(f"\nLeaderboard metrics saved to {metrics_out}")

else:
    print(
        f"\n[!] Annotations file not found at {ANNOTATIONS_FILE}. Skipping evaluation."
    )
