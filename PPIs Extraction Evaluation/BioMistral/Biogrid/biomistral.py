import pandas as pd
import os

# === Parameters ===
output_dir = "/data/gent/490/vsc49096/results"
os.makedirs(output_dir, exist_ok=True)

llm_file = r"/data/gent/490/vsc49096/all_ppi_predictions_BioMistral.csv"
benchmark_file = r"/data/gent/490/vsc49096/clean_BIOGRID.tsv"
model_name = "biomistral"
chunk_size = 10_000  # adjust if needed

# Output CSVs
true_positives_file = os.path.join(output_dir, f"{model_name}_true_positives.csv")
false_positives_file = os.path.join(output_dir, f"{model_name}_false_positives.csv")

# --- Helper to split protein names ---
def split_names(cell):
    if pd.isna(cell):
        return []
    return [name.strip().lower() for name in str(cell).split(",") if name.strip()]

# --- Step 1: Build benchmark pairs (all synonym combinations) ---
print("Building benchmark lookup dictionary...")
benchmark_pairs = set()
for chunk in pd.read_csv(benchmark_file, sep=",", usecols=['source','target'], chunksize=chunk_size):
    for _, row in chunk.iterrows():
        source_list = split_names(row['source'])
        target_list = split_names(row['target'])
        for s in source_list:
            for t in target_list:
                if s != "" and t != "" and s != t:
                    benchmark_pairs.add(tuple(sorted([s, t])))

print(f"Benchmark contains {len(benchmark_pairs)} unique interaction pairs.")

# --- Step 2: Process LLM predictions ---
true_positives_count = 0
total_predictions_count = 0

# Reset output CSVs
for f in [true_positives_file, false_positives_file]:
    if os.path.exists(f):
        os.remove(f)

for chunk in pd.read_csv(llm_file, chunksize=chunk_size):
    chunk["protein_1"] = chunk["protein_1"].astype(str).str.lower()
    chunk["protein_2"] = chunk["protein_2"].astype(str).str.lower()

    true_rows, false_rows = [], []

    for _, row in chunk.iterrows():
        pair = tuple(sorted([row["protein_1"], row["protein_2"]]))
        if pair in benchmark_pairs:
            true_rows.append(row)
            true_positives_count += 1
        else:
            false_rows.append(row)

    total_predictions_count += len(chunk)

    # Save incremental results
    if true_rows:
        pd.DataFrame(true_rows).to_csv(true_positives_file, mode='a', index=False,
                                       header=not os.path.exists(true_positives_file))
    if false_rows:
        pd.DataFrame(false_rows).to_csv(false_positives_file, mode='a', index=False,
                                        header=not os.path.exists(false_positives_file))

# --- Step 3: Compute precision ---
false_positives_count = total_predictions_count - true_positives_count
precision = true_positives_count / (true_positives_count + false_positives_count) \
    if (true_positives_count + false_positives_count) > 0 else 0

# --- Step 4: Save summary ---
summary = pd.DataFrame([{
    "model": model_name,
    "true_positives": true_positives_count,
    "false_positives": false_positives_count,
    "total_predictions": total_predictions_count,
    "precision": round(precision, 4)
}])

summary.to_csv(os.path.join(output_dir, f"{model_name}_metrics_summary.csv"), index=False)

print(f"\n✅ Evaluation completed for {model_name}")
print(summary.to_string(index=False))
print(f"True positives saved to {true_positives_file}")
print(f"False positives saved to {false_positives_file}")
