import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from tqdm import tqdm

from constants import REDDIT_LANGUAGE

# ==========================================
# CONSENSUS CONFIGURATION
# ==========================================
# Minimum number of models that must vote "1" for a row to count as a strong
# RELEVANT consensus. The IRRELEVANT consensus cutoff is derived symmetrically
# from this (num_models - CONSENSUS_THRESHOLD). Adjust this whenever the
# number of models changes.
CONSENSUS_THRESHOLD = 3

script_dir = Path(__file__).parent
results_dir = script_dir / "artifacts" / "results" / f"reddit-{REDDIT_LANGUAGE}" / "llm"

all_data = []

# Load all ndjson files
for file_path in tqdm(results_dir.glob("*.ndjson"), desc="Loading ndjson files"):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

df = pd.DataFrame(all_data)

# Clean up: Force labels to strings and strip whitespace to ensure "1" == " 1"
df["label"] = df["label"].astype(str).str.strip()

# Pivot the data: remove duplicate entries for the same id and model, keeping the first occurrence
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
# Symmetric cutoff: with N models, needing >= CONSENSUS_THRESHOLD votes for
# RELEVANT means the mirror-image IRRELEVANT cutoff is <= (N - CONSENSUS_THRESHOLD) votes.
IRRELEVANT_CONSENSUS_THRESHOLD = len(model_cols) - CONSENSUS_THRESHOLD

# Count how many models predicted "1" (relevant) for each row (axis=1 sums across columns for each row)
agreement_series = (df_final[model_cols] == "1").sum(axis=1)
# Count the total number of rows with strong consensus on 'relevant'
relevant_consensus = (agreement_series >= CONSENSUS_THRESHOLD).sum()
# Count the total number of rows with strong consensus on 'irrelevant'
irrelevant_consensus = (agreement_series <= IRRELEVANT_CONSENSUS_THRESHOLD).sum()

print(
    f"Items with >={CONSENSUS_THRESHOLD} models agreeing on RELEVANT: {relevant_consensus}"
)
print(
    f"Items with <={IRRELEVANT_CONSENSUS_THRESHOLD} models agreeing on RELEVANT (i.e. IRRELEVANT consensus): {irrelevant_consensus}"
)
print(f"Total High-Confidence Items: {relevant_consensus + irrelevant_consensus}")

# 1. Calculate the number of '1' votes per row
model_cols = [c for c in df_final.columns if c not in ["id", "text"]]
agreement_series = (df_final[model_cols] == "1").sum(axis=1)

# 2. 'No Agreement' is any vote count strictly between the two consensus thresholds
no_agreement_mask = (agreement_series > IRRELEVANT_CONSENSUS_THRESHOLD) & (
    agreement_series < CONSENSUS_THRESHOLD
)
no_agreement_count = no_agreement_mask.sum()

print(f"Total rows with no strong agreement: {no_agreement_count}")
print(
    f"Percentage of dataset in conflict: {(no_agreement_count / len(df_final)) * 100:.2f}%"
)

# 3. Peek at the 'messy' data to see why they disagree
print("\nExamples of high-conflict rows:")
print(df_final[no_agreement_mask][["text"] + model_cols].head())

# df_final.to_csv("master_llm_consensus.csv", index=False)

# ==========================================
# ENSEMBLE VOTING HELPERS
# ==========================================


def _majority_vote(
    labels: list[str],
    tie_break_priority: list[str] | None = None,
) -> str:
    """Return the majority label from a list of predictions.

    When there is a tie, the label appearing first in *tie_break_priority* wins.
    If no priority list is given, the alphabetically first label is chosen so
    the result is deterministic.
    """
    counts = Counter(labels)
    max_count = max(counts.values())
    tied = [label for label, count in counts.items() if count == max_count]
    if len(tied) == 1:
        return tied[0]
    if tie_break_priority:
        for priority_label in tie_break_priority:
            if priority_label in tied:
                return priority_label
    return sorted(tied)[0]


def _add_ensemble_columns(
    df: pd.DataFrame,
    model_cols: list[str],
    tie_break_priority: list[str] | None = None,
) -> list[str]:
    """Add ensemble majority-vote columns to *df* in-place.

    Generates:
    1. ``ensemble::all (N)`` – vote across all models.
    2. ``ensemble::top-K`` for K in {3, 5} – vote across the top-K models
       (by column order, which mirrors the leaderboard sort the caller
       performs).
    3. Every combination of 3 or more models – useful for finding the best
       possible ensemble.

    Returns the list of newly created column names.
    """
    if len(model_cols) < 2:
        return []

    priority = tie_break_priority or ["0", "1"]
    new_cols: list[str] = []

    # --- Full ensemble across all models -----------------------------------
    full_col = f"ensemble::all ({len(model_cols)})"
    df[full_col] = df.apply(
        lambda row: _majority_vote(
            [row[c] for c in model_cols if pd.notna(row[c])],
            tie_break_priority=priority,
        )
        if any(pd.notna(row[c]) for c in model_cols)
        else None,
        axis=1,
    )
    new_cols.append(full_col)

    # --- Top-K ensembles ---------------------------------------------------
    for k in (3, 5):
        if k < len(model_cols):
            top_k_cols = model_cols[:k]
            col_name = f"ensemble::top-{k}"
            df[col_name] = df.apply(
                lambda row, _cols=top_k_cols: _majority_vote(
                    [row[c] for c in _cols if pd.notna(row[c])],
                    tie_break_priority=priority,
                )
                if any(pd.notna(row[c]) for c in _cols)
                else None,
                axis=1,
            )
            new_cols.append(col_name)

    # --- All combinations of 3+ models ------------------------------------
    if len(model_cols) <= 12:  # cap to avoid combinatorial explosion
        for r in range(3, len(model_cols)):
            for combo in combinations(model_cols, r):
                combo_slugs = [c.split("/")[-1] for c in combo]
                col_name = f"ensemble::combo({'+'.join(combo_slugs)})"
                if col_name in new_cols:
                    continue
                combo_list = list(combo)
                df[col_name] = df.apply(
                    lambda row, _cols=combo_list: _majority_vote(
                        [row[c] for c in _cols if pd.notna(row[c])],
                        tie_break_priority=priority,
                    )
                    if any(pd.notna(row[c]) for c in _cols)
                    else None,
                    axis=1,
                )
                new_cols.append(col_name)

    return new_cols


ANNOTATIONS_FILE = (
    script_dir / "annotations" / f"reddit_{REDDIT_LANGUAGE}_manual_eval_labels.json"
)

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

    # Sort model columns by individual F1 (descending) so that top-K
    # ensembles use the strongest models.
    individual_f1: dict[str, float] = {}
    for result in eval_results:
        individual_f1[str(result["Model"])] = float(result["F1-Score (1)"])

    model_cols_sorted = sorted(
        model_cols,
        key=lambda c: individual_f1.get(c, 0.0),
        reverse=True,
    )

    # --- Build and evaluate ensemble columns -------------------------------
    # Tie-break priority: prefer "0" (irrelevant) on ties for a conservative
    # (precision-first) approach.
    ensemble_cols = _add_ensemble_columns(
        df_eval, model_cols_sorted, tie_break_priority=["0", "1"]
    )

    if ensemble_cols:
        print(f"\n{'=' * 60}")
        print(f" EVALUATING {len(ensemble_cols)} ENSEMBLE VOTING COMBINATION(S) ")
        print("=" * 60)

        for ens_col in ensemble_cols:
            y_pred = (
                pd.to_numeric(df_eval[ens_col], errors="coerce")
                .fillna(-1)
                .astype(int)
            )
            y_true = df_eval["gold_label"]

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            kappa = cohen_kappa_score(y_true, y_pred)
            accuracy = (tp + tn) / len(df_eval) if len(df_eval) > 0 else 0
            precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_val = (
                2 * (precision_val * recall_val) / (precision_val + recall_val)
                if (precision_val + recall_val) > 0
                else 0
            )

            eval_results.append(
                {
                    "Model": ens_col,
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

    # Show a focused ensemble-only view so the comparison is easy to spot.
    ensemble_rows = df_metrics[
        df_metrics.index.str.startswith("ensemble::")
    ]
    if not ensemble_rows.empty:
        print("\n" + "=" * 60)
        print(" ENSEMBLE-ONLY LEADERBOARD (Sorted by F1-Score) ")
        print("=" * 60)
        print(
            ensemble_rows[
                ["Accuracy", "Precision (1)", "Recall (1)", "F1-Score (1)", "Cohen's Kappa"]
            ].to_string()
        )

    # Save summary metrics to disk
    metrics_out = (
        script_dir
        / "artifacts"
        / "results"
        / f"reddit-{REDDIT_LANGUAGE}"
        / "classification-evaluation.csv"
    )
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(metrics_out)
    print(f"\nLeaderboard metrics saved to {metrics_out}")

else:
    print(
        f"\n[!] Annotations file not found at {ANNOTATIONS_FILE}. Skipping evaluation."
    )

