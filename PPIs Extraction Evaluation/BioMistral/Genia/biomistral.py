import pandas as pd
import os

# === Parameters ===
output_dir = r"C:\Users\olive\Desktop\UGent\Thesis\Pratice\Final Stages\BioMistral\Genia"
os.makedirs(output_dir, exist_ok=True)

llm_file = r"C:\Users\olive\Desktop\UGent\Thesis\Pratice\PPIs\Model Testing 2\BioMistral\all_ppi_predictions_BioMistral.csv"
benchmark_file = r"C:\Users\olive\Desktop\UGent\Thesis\Pratice\genia\GENIA_event_annotation_0.9\genia_interactions.csv"  # GENIA format
model_name = "biomistral"
chunk_size = 10_000

# Output CSVs
true_positives_file = os.path.join(output_dir, f"{model_name}_true_positives.csv")
false_positives_file = os.path.join(output_dir, f"{model_name}_false_positives.csv")
false_negatives_file = os.path.join(output_dir, f"{model_name}_false_negatives.csv")

# --- Step 1: Load GENIA benchmark ---
print("Loading GENIA benchmark...")
genia = pd.read_csv(benchmark_file)

# === Function to expand synonyms ===
def expand_synonyms(cell):
    """Split a benchmark protein cell into possible synonyms (only by comma)."""
    if pd.isna(cell):
        return []
    parts = [x.strip().lower() for x in str(cell).split(",") if x.strip()]
    return parts

# Expand benchmark into all synonym pairs
benchmark_pairs = []
for _, row in genia.iterrows():
    proteins_A = expand_synonyms(row["protein_A"])
    proteins_B = expand_synonyms(row["protein_B"])
    for a in proteins_A:
        for b in proteins_B:
            benchmark_pairs.append(tuple(sorted([a, b])))

benchmark_set = set(benchmark_pairs)
print(f"Benchmark expanded to {len(benchmark_set)} protein pairs (with synonyms).")

# --- Step 2: Evaluate predictions ---
true_positives_count = 0
total_predictions_count = 0
predicted_pairs = set()

# Reset outputs
for f in [true_positives_file, false_positives_file, false_negatives_file]:
    if os.path.exists(f):
        os.remove(f)

for chunk in pd.read_csv(llm_file, chunksize=chunk_size):
    chunk["protein_1"] = chunk["protein_1"].astype(str).str.lower()
    chunk["protein_2"] = chunk["protein_2"].astype(str).str.lower()

    true_rows, false_rows = [], []

    for _, row in chunk.iterrows():
        p1, p2 = row["protein_1"], row["protein_2"]
        pair = tuple(sorted([p1, p2]))
        predicted_pairs.add(pair)

        if pair in benchmark_set:
            true_rows.append(row)
            true_positives_count += 1
        else:
            false_rows.append(row)

    total_predictions_count += len(chunk)

    # Save incrementally
    if true_rows:
        pd.DataFrame(true_rows).to_csv(true_positives_file, mode='a', index=False,
                                       header=not os.path.exists(true_positives_file))
    if false_rows:
        pd.DataFrame(false_rows).to_csv(false_positives_file, mode='a', index=False,
                                        header=not os.path.exists(false_positives_file))

# --- Step 3: False negatives ---
false_negatives = benchmark_set - predicted_pairs
false_negatives_count = len(false_negatives)

if false_negatives:
    pd.DataFrame(false_negatives, columns=["protein_A", "protein_B"]).to_csv(
        false_negatives_file, index=False
    )

# --- Step 4: Metrics ---
false_positives_count = total_predictions_count - true_positives_count

precision = true_positives_count / (true_positives_count + false_positives_count) if (true_positives_count + false_positives_count) > 0 else 0
recall = true_positives_count / (true_positives_count + false_negatives_count) if (true_positives_count + false_negatives_count) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# --- Step 5: Save summary ---
summary = pd.DataFrame([{
    "model": model_name,
    "true_positives": true_positives_count,
    "false_positives": false_positives_count,
    "false_negatives": false_negatives_count,
    "total_predictions": total_predictions_count,
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1_score, 4)
}])

summary.to_csv(os.path.join(output_dir, f"{model_name}_metrics_summary.csv"), index=False)

print(f"\n✅ Evaluation completed for {model_name} on GENIA")
print(summary.to_string(index=False))
print(f"True positives saved to {true_positives_file}")
print(f"False positives saved to {false_positives_file}")
print(f"False negatives saved to {false_negatives_file}")
