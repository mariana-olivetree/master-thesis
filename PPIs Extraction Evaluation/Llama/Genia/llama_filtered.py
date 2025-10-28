import pandas as pd
import os

# === Parameters ===
output_dir = r"\Users\olive\Desktop\UGent\Thesis\Pratice\Final Stages\Llama\Genia"
os.makedirs(output_dir, exist_ok=True)

llm_file = r"\Users\olive\Desktop\UGent\Thesis\Pratice\PPIs\Model Testing 2\Llama13\all_ppi_predictions_llama.csv"
benchmark_file = r"\Users\olive\Desktop\UGent\Thesis\Pratice\genia\GENIA_event_annotation_0.9\filtered_data.csv"  # GENIA format
model_name = "llama"
chunk_size = 10_000

# Output CSVs
true_positives_file = os.path.join(output_dir, f"{model_name}_true_positives.csv")
false_positives_file = os.path.join(output_dir, f"{model_name}_false_positives.csv")

# --- Step 1: Load GENIA benchmark ---
print("Loading GENIA benchmark...")
genia = pd.read_csv(benchmark_file)

# Normalize to lowercase
genia["protein_A"] = genia["protein_A"].astype(str).str.lower()
genia["protein_B"] = genia["protein_B"].astype(str).str.lower()

benchmark_records = genia[["protein_A", "protein_B"]].values.tolist()
print(f"Benchmark contains {len(benchmark_records)} interaction entries (sentences).")

# --- Step 2: Evaluate predictions ---
true_positives_count = 0
total_predictions_count = 0

# Reset outputs
for f in [true_positives_file, false_positives_file]:
    if os.path.exists(f):
        os.remove(f)

def is_match(p1, p2, protA, protB):
    """Check if prediction proteins occur in either order in the GENIA sentence pairs."""
    return (p1 in protA and p2 in protB) or (p1 in protB and p2 in protA)

for chunk in pd.read_csv(llm_file, chunksize=chunk_size):
    chunk["protein_1"] = chunk["protein_1"].astype(str).str.lower()
    chunk["protein_2"] = chunk["protein_2"].astype(str).str.lower()

    true_rows, false_rows = [], []

    for _, row in chunk.iterrows():
        p1, p2 = row["protein_1"], row["protein_2"]

        match_found = False
        for protA, protB in benchmark_records:
            if is_match(p1, p2, protA, protB):
                true_rows.append(row)
                true_positives_count += 1
                match_found = True
                break

        if not match_found:
            false_rows.append(row)

    total_predictions_count += len(chunk)

    # Save incrementally
    if true_rows:
        pd.DataFrame(true_rows).to_csv(true_positives_file, mode='a', index=False,
                                       header=not os.path.exists(true_positives_file))
    #if false_rows:
    #   pd.DataFrame(false_rows).to_csv(false_positives_file, mode='a', index=False,
    #                                    header=not os.path.exists(false_positives_file))


# --- Step 3: Metrics ---
false_positives_count = total_predictions_count - true_positives_count
false_negatives_count = len(benchmark_records) - true_positives_count

precision = true_positives_count / (true_positives_count + false_positives_count) \
    if (true_positives_count + false_positives_count) > 0 else 0

recall = true_positives_count / (true_positives_count + false_negatives_count) \
    if (true_positives_count + false_negatives_count) > 0 else 0

f1_score = 2 * (precision * recall) / (precision + recall) \
    if (precision + recall) > 0 else 0

# --- Step 4: Save summary ---
summary = pd.DataFrame([{
    "model": model_name,
    "true_positives": true_positives_count,
    "false_positives": false_positives_count,
    "false_negatives": false_negatives_count,
    "total_predictions": total_predictions_count,
    "benchmark_total": len(benchmark_records),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1_score, 4)
}])

summary.to_csv(os.path.join(output_dir, f"{model_name}_metrics_summary.csv"), index=False)

print(f"\n✅ Evaluation completed for {model_name} on GENIA")
print(summary.to_string(index=False))
#print(f"True positives saved to {true_positives_file}")
#print(f"False positives saved to {false_positives_file}")
