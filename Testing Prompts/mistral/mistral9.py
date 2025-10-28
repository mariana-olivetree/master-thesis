import os
import pandas as pd
import re
import csv

chunks_folder = "chunks_2"  # Your chunks folder
chunks_by_pmid = {}

# Step 1: Load all chunks and group them by PMID
for csv_file in os.listdir(chunks_folder):
    if csv_file.endswith(".csv"):
        pmid = csv_file.split('_')[0]
        csv_path = os.path.join(chunks_folder, csv_file)
        df = pd.read_csv(csv_path)
        df['PMID'] = pmid
        if pmid not in chunks_by_pmid:
            chunks_by_pmid[pmid] = []
        chunks_by_pmid[pmid].append(df)

# Step 2: Concatenate all DataFrames per PMID into one DataFrame
chunks_by_pmid = {pmid: pd.concat(dfs, ignore_index=True) for pmid, dfs in chunks_by_pmid.items()}

# Step 3: Create batches of 2 PDFs
pmid_list = list(chunks_by_pmid.keys())
batch_size = 2

batches = [pmid_list[i:i + batch_size] for i in range(0, len(pmid_list), batch_size)]

top_n = 10  # or however many top chunks you want to retrieve per batch

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch

# Initialize accelerator and get device
accelerator = Accelerator()
device = accelerator.device
model_name = "mistralai/Mistral-7B-v0.1"
token = "your_token"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token,
    cache_dir="/data/gent/490/vsc49096/huggingface"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    cache_dir="/data/gent/490/vsc49096/huggingface",
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# Ensure pad token is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # or any other token you want to use as pad_token

model.config.pad_token_id = tokenizer.pad_token_id

#pip uninstall peft -y
#pip install git+https://github.com/huggingface/peft.git
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

# Load the model on CPU to avoid CUDA memory errors
model_st = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device='cpu')

# Function to encode text on CPU
def embed_text(text, model):
    embeddings = model.encode(text, convert_to_tensor=True, device='cpu')
    return embeddings

query_texts = [
    #1
   """
You are an expert biomedical information extractor.

Task: Extract all protein-protein interactions from the text below in the format:
'Protein A -> [binding/modulation/activation/etc.] -> Protein B'

Only include clear interactions from the main body of the document.

Note: The following example is just for your understanding. Do not include this example or any references to it in your output.

Example (DO NOT INCLUDE THIS IN YOUR OUTPUT):
Input: "AKT1 activates mTOR and binds to GSK3B."
Expected Output:
AKT1 -> activation -> mTOR  
AKT1 -> binding -> GSK3B

---

Now, extract interactions from this context:
""",
#2
"""
From the context below, extract all protein-protein interactions in the format: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.

Note: The following example is just for your understanding. Do not include this example or any references to it in your output.

Example (DO NOT INCLUDE THIS IN YOUR OUTPUT):
Input: "AKT1 activates mTOR and binds to GSK3B."
Expected Output:
AKT1 -> activation -> mTOR  
AKT1 -> binding -> GSK3B  

---

Now process the following context:
""",
#3
"""
From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [Interaction Type] -> Protein B'. Only include clear interactions from the main body of the document.

Note: The following example is just for your understanding. Do not include this example or any references to it in your output.

Example (DO NOT INCLUDE THIS IN YOUR OUTPUT):
Input: "AKT1 activates mTOR and binds to GSK3B."
Expected Output:
AKT1 -> activation -> mTOR  
AKT1 -> binding -> GSK3B  

---

Now process the following context:
""",
#4
"""
Role: You are a biomedical expert specializing in protein interaction extraction.

Task: Extract protein-protein interactions from the text below using the format:
Protein A -> [Interaction Type] -> Protein B

Rules:

Include only explicit interactions from the main document.

NEVER include the example below or any references to it.

Example (For Understanding Only):
Input: "AKT1 activates mTOR and binds to GSK3B."
Output:
AKT1 -> activation -> mTOR
AKT1 -> binding -> GSK3B

Extract interactions from THIS text:
""",
#5
"""
Step 1: Read the text carefully.
Step 2: Identify all protein-protein interactions (e.g., binding, activation).
Step 3: Format each as Protein -> [Interaction] -> Protein.
Step 4: Exclude hypothetical/example content (e.g., AKT1/mTOR/GSK3B).

Example (Do Not Reproduce):
Input: "AKT1 activates mTOR..."
Output: AKT1 -> activation -> mTOR

Process THIS input:
""",
#6
"""
You are an expert extractor. Extract protein interactions in the format Protein A -> [Type] -> Protein B.

Include: Direct interactions from the text (e.g., phosphorylates, binds).

Exclude: Hypotheticals, examples (like AKT1/mTOR/GSK3B), or non-main content.

Never include this example in outputs:
Input: "AKT1 activates mTOR..." → Output: AKT1 -> activation -> mTOR

Target Text:
"""
]

for prompt_index, query_text in enumerate(query_texts, start=1):
    print(f"\n🧪 Testing Prompt {prompt_index}: {query_text}")

    query_embedding = embed_text(query_text, model_st)
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)

    ppi_results = []  # clear previous results

    for batch_num, pmid_batch in enumerate(batches, start=1):
        print(f"\n🔄 Processing batch {batch_num} with PMIDs: {pmid_batch}")

        batch_df = pd.concat([chunks_by_pmid[pmid] for pmid in pmid_batch], ignore_index=True)

        batch_embeddings = np.vstack(
            batch_df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',')).values
        )

        cosine_similarities = cosine_similarity(query_embedding_np, batch_embeddings)

        top_indices = cosine_similarities[0].argsort()[-top_n:][::-1]
        top_chunks = batch_df.iloc[top_indices].copy()
        top_chunks['cosine_similarity'] = cosine_similarities[0][top_indices]

        print(f"Top {top_n} relevant chunks from batch {batch_num}:")
        print(top_chunks[['page_number', 'sentence_chunk', 'cosine_similarity', 'PMID']])

        for i, row in top_chunks.iterrows():
            context = row["sentence_chunk"]
            pmid = row["PMID"]

            prompt = f"Question: {query_text}\nContext: {context}\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    top_k=50,
                    temperature=0.7
                )

            answer = tokenizer.decode(output[0], skip_special_tokens=True)

            print(f"\n📄 PMID: {pmid}, Page: {row['page_number']}")
            print(f"📝 Prompt: {prompt}")
            print(f"🧠 Answer: {answer}")

            matches = re.findall(r'([\w\-]+)\s*->\s*\[?([\w\s\-]+)\]?\s*->\s*([\w\-]+)', answer)
            for match in matches:
                protein_1, interaction_type, protein_2 = match
                ppi_results.append({
                    "PMID": pmid,
                    "protein_1": protein_1,
                    "interaction_type": interaction_type.strip(),
                    "protein_2": protein_2,
                    "context": context
                })

    # Save CSV after all batches are processed for this prompt
    output_file = f"ppi_predictions_prompt_{prompt_index}.csv"
    os.makedirs("ppi_outputs_mistral8", exist_ok=True)
    output_path = os.path.join("ppi_outputs_mistral8", output_file)

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["PMID", "protein_1", "interaction_type", "protein_2", "context"])
        writer.writeheader()
        writer.writerows(ppi_results)

    print(f"✅ Saved results for Prompt {prompt_index} to {output_path}")