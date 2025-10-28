import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import re
import csv
import numpy as np


# Now import transformers AFTER handling the Replicate issue
from accelerate import Accelerator

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


########################
#cache_dir = "/data/gent/490/vsc49096/huggingface"
#token = "your_token"  # Replace with your real token

accelerator = Accelerator()
device = accelerator.device

# Use direct path to model directory (not snapshot subfolder)
model_path = "/scratch/gent/490/vsc49096/meditron-7b-gptq"

# Load model and tokenizer ⚠️ USING args.model_path
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,  # ⚠️ USE SAME PATH
    local_files_only=True
)

# Pad token setup ⚠️ MOVED EARLIER
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
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
    
    ### Mistral 1.1
    """From the context below, extract all protein-protein interactions in the format: 'Protein A -> [interaction type] -> Protein B'. Only include clear interactions from the main body of the document.""",

    ### Mistral 1.2
    """Identify any protein interactions mentioned and format them as: 'Protein A -> [interaction] -> Protein B'.""",

    ### Mistral 1.3
    """List all protein-protein relations in the following form: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'.""",

    ### Mistral 1.4
    """From the given context, return protein interaction pairs with interaction type in this format: 'Protein A -> [action] -> Protein B'.""",

    ### Mistral 1.5
    """Extract and format any explicit protein-protein interactions found using this template: 'Protein A -> [interaction type] -> Protein B'.""",

    ### Mistral 2.1
    """You are a biomedical research assistant specializing in extracting protein-protein interactions from scientific articles. Only extract information that is directly stated in the text. Do not invent interactions and only list what the text supports. From the text below, identify and extract all explicitly stated protein-protein interactions.
    Output a list of interactions in the following structured format: 'Protein [name] -> [Interaction type if available] -> Protein [name]'""",

    ### Mistral 2.2 MODIFICADA
    """You are tasked with extracting protein-protein interactions from biomedical literature. Focus on accuracy and clear evidence from the text. Structure your output as follows: 'Protein [name] -> [Interaction type if available] -> Protein [name]'""",

    ### Mistral 2.3 MODIFICADA
    """You are a biomedical research assistant specializing in extracting protein-protein interactions from scientific articles. From the text below, identify and extract all explicitly stated protein-protein interactions. Output a list of interactions in the following structured format: 'Protein [name] -> [Interaction type if available] -> Protein [name]'.
    
    Example 1:
    Text: "The interaction between p53 and MDM2 is crucial for regulating the cell cycle. p53 binds to MDM2 to inhibit its activity."
    Extraction: Protein p53 -> inhibits -> Protein MDM2
    
    Example 2
    Text: "EGFR forms a complex with GRB2 following activation by EGF."
    Extraction: Protein EGFR -> complex formation -> Protein GRB2

    Process the text below:
    """,

    ### Mistral 2.4 MODIFICADA
    """You are tasked with extracting protein-protein interactions from biomedical literature. Focus on accuracy and clear evidence from the text. Structure your output as follows: 'Protein [name] -> [Interaction type if available] -> Protein [name]'.

    Example 1:
    Text: "Upon DNA damage, ATM phosphorylates and activates CHEK2, promoting checkpoint arrest."
    Extraction: Protein ATM -> activates via phosphorylation -> Protein CHEK2
   
    Example 2:
    Text: "Studies suggest that MYC may cooperate with MAX to drive transcriptional activation, although their interaction was not directly tested."
    Extraction: Protein MYC -> cooperates -> Protein MAX
    
    Extract the protein-protein interactions from the following context:
    """,

    ### Mistral 3
    """From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.""",

    ### Mistral 4 MODIFICADA
    """
    Consider each sentence separately and infer every pair of Protein-Protein Interactions from the provided sentences. 
    For this task, consider Proteins and Genes as interchangeable terms. 
    Provide each pair in a separate row whenever a sentence contains multiple Protein-Protein interaction pairs. 
    Please, format your results in the following format: 'Protein A -> [interaction type] -> Protein B'.
    Output Specifications:
    'Protein 1' and 'Protein 2': The entities in the sentence, representing the proteins or genes.
    'Interaction Type': The type of interaction identified between the protein entities (e.g., 'binds to', 'inhibits').
    Here are the sentences that you need to process:
    """,

    ### Mistral 5.1 MODIFICADA
    """
    For each sentence below, perform the following steps:
    1. Identify all proteins or genes (treat as interchangeable).
    2. For every pair of proteins/genes, determine if an interaction is described.
    3. For each interaction, specify the type (e.g., 'binds to', 'activates', 'inhibits').
    4. Output results in the following format: 'Protein A -> [interaction type] -> Protein B'.
    5. Each row should represent one interaction pair.

    Input:
    Sentence

    Example:
    "EGFR phosphorylates STAT3 and interacts with GRB2."
    "MYC represses the expression of CDKN1A."

    Expected output:
    EGFR -> phosphorylates -> STAT3,
    EGFR -> interacts with -> GRB2
    MYC -> represses -> CDKN1A

    Process the following sentences:
    """,

    ### Mistral 5.2 É IGUAL À ANTERIOR
    """
    For each sentence below, perform the following steps:
    1. Identify all proteins or genes (treat as interchangeable).
    2. For every pair of proteins/genes, determine if an interaction is described.
    3. For each interaction, specify the type (e.g., 'binds to', 'activates', 'inhibits').
    4. Output results in the following format: 'Protein A -> [interaction type] -> Protein B'.
    5. Each row should represent one interaction pair.

    Input:
    Sentence

    Example:
    "EGFR phosphorylates STAT3 and interacts with GRB2."
    "MYC represses the expression of CDKN1A."

    Expected output:
    EGFR -> phosphorylates -> STAT3,
    EGFR -> interacts with -> GRB2,
    MYC -> represses -> CDKN1A

    Process the following sentences:
    """,

    ### Mistral 5.3 MODIFICADA
    """
    Extract all protein-protein (or gene-gene) interactions from each sentence below. For each detected interaction, output in the following format: 'Protein A -> [interaction type] -> Protein B'.

    Rules:
    - Treat 'protein' and 'gene' as synonyms.
    - If multiple interactions are present in a sentence, output each as a separate.
    - Do not leave any blank fields.

    Input format: Sentence

    Example input:
    "AKT1 activates mTOR and binds to GSK3B."
    "TP53 forms a complex with BAX."

    Example output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B
    TP53 -> forms complex with -> BAX5

    Sentences:
    """,

    ### Mistral 6.1
    """
    Extract all protein-protein or gene-gene interactions from the following text. For each interaction, write one line in the format: 'Protein1 -> [Interaction Type] -> Protein2'. If there are multiple interactions, list each on a separate line.
    """,

    ### Mistral 6.2 MODIFICADA
    """
    You are an expert in biomedical text mining.
    Given the following text, identify every protein-protein or gene-gene interaction described.
    Treat "protein" and "gene" as interchangeable.
    For each interaction, output a single line in this exact format: 'Protein1 -> [Interaction Type] -> Protein2'
    """,

    ### Mistral 6.3 MODIFICADA
    """
    Extract all protein-protein or gene-gene interactions from the following text.
    For each interaction, write one line in the format: 'Protein1 -> [Interaction Type] -> Protein2'

    Examples:
    Text: "p53 binds to MDM2 and inhibits its activity."
    Output:
    p53 -> binds to -> MDM2
    p53 -> inhibits -> MDM2

    Text: "BRCA1 interacts with RAD51 during DNA repair."
    Output:
    BRCA1 -> interacts with -> RAD51

    Text:
    """,

    ### Mistral 7.1

    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.

    Example input:
    "AKT1 activates mTOR and binds to GSK3B."

    Example output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    Context:
    """,

    ### Mistral 7.2
    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.

    Example input:
    "AKT1 activates mTOR and binds to GSK3B."

    Example output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    Process the following context:
    """,

    ### Mistral 7.3

    """Extract all protein-protein interactions from the context below in the format 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.""",

    ### Mistral 7.4

    """
    Extract all protein-protein interactions from the context below in the format 'Protein A ->  [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.

    Example input:
    "AKT1 activates mTOR and binds to GSK3B."

    Example output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    Context:
    """,

    ### Mistral 7.5

    """
    Extract all protein-protein interactions from the context below in the format 'Protein A ->  [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.

    Example input:
    "AKT1 activates mTOR and binds to GSK3B."

    Example output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    Process the following context:
    """,

    ### Mistral 7.6
    """
    For each sentence below, perform the following steps: identify all proteins; for every pair of proteins, determine if an interaction is described; for each interaction, specify the type (e.g., 'binds to', 'activates', 'inhibits'); extract results in the format 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'.

    Example:
    "EGFR phosphorylates STAT3 and interacts with GRB2."

    Expected output:
    EGFR -> phosphorylates -> STAT3
    EGFR -> interacts with -> GRB2


    Process the following sentences:
    """,

    ### Mistral 7.7    

    """
    Infer every pair of Protein-Protein Interactions from the provided context. Provide each pair in a separate row whenever a sentence contains multiple Protein-Protein interaction pairs. Please, format your results in the format 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Ensure that no columns are left blank.
    Output Column Specifications:
    'Protein A' and 'Protein B': The entities in the sentence, representing the proteins.
    'Interaction Type': The type of interaction identified between the protein entities (e.g., 'binds to', 'inhibits'). 
    Here are the sentences that you need to process:
    """,

    ### Mistral 8.1
    "From the context below, extract all protein-protein interactions in the format: 'Protein A -> [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.",

    ### Mistral 8.2
    "From the context below, extract all protein-protein interactions in the format: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.",

    ### Mistral 8.3
    "List all protein-protein relations in the following form: 'Protein A -> [interaction type] -> Protein B'.",

    ### Mistral 8.4
    "List all protein-protein relations in the following form: 'Protein A -> [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'.",

    ### Mistral 8.5
    "List all protein-protein relations in the following form: 'Protein A -> [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.",
     
    ### Mistral 8.6
    "From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.",

    ### Mistral 8.7
    "From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [Interaction Type] -> Protein B'. Only include clear interactions from the main body of the document.",

    ### Mistral 8.8
    """From the context below, extract all protein-protein interactions in the format: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.

    Example:

    Input: "AKT1 activates mTOR and binds to GSK3B."
    Output:
    AKT1 -> activation -> mTOR  
    AKT1 -> binding -> GSK3B  

    ---

    Now process the following context:
    """,

    ### Mistral 8.9

    """
    From the context below, extract all protein-protein interactions in the format: 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document. Use the following example for guidance but never output them.

    Example:

    Input: "AKT1 activates mTOR and binds to GSK3B."
    Output:
    AKT1 -> activation -> mTOR  
    AKT1 -> binding -> GSK3B  


    Process the following context:
    """,

    ### Mistral 8.10

    """
    Extract all protein-protein interactions from the context below in the format 'Protein A ->  [Interaction Type (e.g., 'binds to', 'activates', 'forms complex with')] -> Protein B'. Only include clear interactions from the main body of the document.

    Example:

    Input: "AKT1 activates mTOR and binds to GSK3B."
    Output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    ---

    Now process the following context:
    """,

    ### Mistral 8.11
    """
    Extract all protein-protein interactions from the context below in the format 'Protein A ->  [binding/modulation/activation/etc.] -> Protein B'. Only include clear interactions from the main body of the document.

    Example:

    Input: "AKT1 activates mTOR and binds to GSK3B."
    Output:
    AKT1 -> activates -> mTOR
    AKT1 -> binds to -> GSK3B

    ---

    Now process the following context:
    """,

    ### Mistral 9.1

    """
    You are an expert biomedical information extractor.

    Task: Extract all protein-protein interactions from the text below in the format: 'Protein A -> [binding/modulation/activation/etc.] -> Protein B'

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

    ### Mistral 9.2

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

    ### Mistral 9.3
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

    ### Mistral 9.4
    """    
    Role: You are a biomedical expert specializing in protein interaction extraction.

    Task: Extract protein-protein interactions from the text below using the format: 'Protein A -> [Interaction Type] -> Protein B'.

    Rules: Include only explicit interactions from the main document and NEVER include the example below or any references to it.

    Example (For Understanding Only):

    Input: "AKT1 activates mTOR and binds to GSK3B."
    Output:
    AKT1 -> activation -> mTOR
    AKT1 -> binding -> GSK3B

    Extract interactions from THIS text:
    """,

    ### Mistral 9.5

    """
    Step 1: Read the text carefully.
    Step 2: Identify all protein-protein interactions (e.g., binding, activation).
    Step 3: Format each as: 'Protein -> [Interaction] -> Protein'.
    Step 4: Exclude hypothetical/example content (e.g., AKT1/mTOR/GSK3B).

    Example (Do Not Reproduce):

    Input: "AKT1 activates mTOR..."
    Output: AKT1 -> activation -> mTOR

    Process THIS input:
    """,

    #Mistral 9.6
    """
    You are an expert extractor. Extract protein interactions in the format: 'Protein A -> [Type] -> Protein B'.

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
    os.makedirs("ppi_outputs_meditron", exist_ok=True)
    output_path = os.path.join("ppi_outputs_meditron", output_file)

    with open(output_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=["PMID", "protein_1", "interaction_type", "protein_2", "context"])
        writer.writeheader()
        writer.writerows(ppi_results)

    print(f"✅ Saved results for Prompt {prompt_index} to {output_path}")