import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re
import csv
import numpy as np


os.environ["HF_HOME"] = os.path.join(os.environ["VSC_SCRATCH"], "huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["VSC_SCRATCH"], "huggingface")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(os.environ["VSC_SCRATCH"], "huggingface")


# Now import transformers AFTER handling the Replicate issue
from accelerate import Accelerator

chunks_folder = "/data/gent/490/vsc49096/chunks"  # Your chunks folder
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
batch_size = 1

batches = [pmid_list[i:i + batch_size] for i in range(0, len(pmid_list), batch_size)]

top_n = 10  # or however many top chunks you want to retrieve per batch


########################
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import Accelerator
import torch

# Initialize accelerator and get device
accelerator = Accelerator()
device = accelerator.device
model_name = "TheBloke/meditron-7B-GPTQ"
token = "your_token"


tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token,
    cache_dir="/scratch/gent/490/vsc49096/meditron-7B-GPTQ"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    cache_dir="/scratch/gent/490/vsc49096/meditron-7B-GPTQ",
    device_map="auto",
    torch_dtype=torch.float16
)

# Ensure pad token is set correctly
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # or any other token you want to use as pad_token

model.config.pad_token_id = tokenizer.pad_token_id
# Ensure pad token is set correctly

#############################

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

# Load the model on CPU to avoid CUDA memory errors
model_st = SentenceTransformer('intfloat/multilingual-e5-large-instruct', device='cpu')

# Function to encode text on CPU
def embed_text(text, model):
    embeddings = model.encode(text, convert_to_tensor=True, device='cpu')
    return embeddings

query_texts = [
## 3
    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.
    """,

    ## 6
    """
    You are a biomedical research assistant specializing in extracting protein-protein interactions from scientific articles. Only extract information that is directly stated in the text. Do not invent interactions and only list what the text supports. From the text below, identify and extract all explicitly stated protein-protein interactions.
    Output a list of interactions in the following structured format: 
    "
    Protein [name] -> [Interaction type if available] -> Protein [name]
    Evidence: [direct quotation from text]
    "
    """,

    ## 7
    """
    You are tasked with extracting protein-protein interactions from biomedical literature. For each interaction you extract, assign a confidence score based on how explicitly the interaction is described (High, Medium, Low). Focus on accuracy and clear evidence from the text. Structure your output as follows:
    "
    Protein [name] -> [Interaction type if available] -> Protein [name]
    Confidence Level: [high/medium/low]
    Evidence: [direct quotation from text]
    "
    """,

    ## 22
    """
    Extract all protein-protein interactions from the context below in the format 'Protein A ->  [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.

    Example input:
    "AKT1 activates mTOR and binds to GSK3B."

    Example output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    Process the following context:
    """,

    ## 25
    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A -> [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.
    """,

    ## 26
    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.
    """,

    ## 29
    """
    List all protein-protein relations in the following form: 'Protein A -> [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.
    """,

    ## 30
    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.
    """,

    ## 32
    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.

    Example:
    Input: "AKT1 activates mTOR and binds to GSK3B."
    Output:
    AKT1 -> activation -> mTOR  
    AKT1 -> binding -> GSK3B  

    ---

    Now process the following context:
    """,

    ## 36 
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
    """
]

ppi_results = []  # Collect results across all prompts

for prompt_index, query_text in enumerate(query_texts, start=1):
    print(f"\n🧪 Testing Prompt {prompt_index}: {query_text}")

    query_embedding = embed_text(query_text, model_st)
    query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)

    for pmid in pmid_list:  # Process one article (PMID) at a time
        print(f"\n📄 Processing article with PMID: {pmid}")
        df = chunks_by_pmid[pmid]

        # Reconstruct embeddings
        embeddings = np.vstack(
            df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=',')).values
        )

        # Compute cosine similarity per chunk in this article
        cosine_similarities = cosine_similarity(query_embedding_np, embeddings)

        # Get top N chunks *within this article*
        top_indices = cosine_similarities[0].argsort()[-top_n:][::-1]
        top_chunks = df.iloc[top_indices].copy()
        top_chunks['cosine_similarity'] = cosine_similarities[0][top_indices]

        print(f"Top {top_n} relevant chunks for PMID {pmid}:")
        print(top_chunks[['page_number', 'sentence_chunk', 'cosine_similarity']])

        for i, row in top_chunks.iterrows():
            context = row["sentence_chunk"]

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
                    "Prompt": str(prompt_index),
                    "protein_1": protein_1,
                    "interaction_type": interaction_type.strip(),
                    "protein_2": protein_2,
                    "context": context                    
                })


# Save one CSV per PMID
output_dir = "ppi_outputs"
os.makedirs(output_dir, exist_ok=True)

# Group results by PMID
from collections import defaultdict

results_by_pmid = defaultdict(list)
for row in ppi_results:
    results_by_pmid[row["PMID"]].append(row)

for pmid, rows in results_by_pmid.items():
    output_file = os.path.join(output_dir, f"ppi_predictions_{pmid}.csv")
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["PMID", "Prompt", "protein_1", "interaction_type", "protein_2", "context"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved results for PMID {pmid} to {output_file}")

