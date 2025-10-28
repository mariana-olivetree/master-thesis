from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator
import torch


# --- Add this error handling block here ---
try:
    from torch.distributed.tensor.parallel import Replicate
except ImportError as e:
    print(f"Warning: Could not import Replicate - {str(e)}")
    print("Attempting fallback...")
    # You might need to define a dummy Replicate class if absolutely necessary
    class Replicate:
        pass


# Set up Hugging Face access
model_name = "meta-llama/Llama-2-7b-hf"
token = "your_token"  # Replace with your real token

# Cache directory in your HPC
cache_dir = "/data/gent/490/vsc49096/huggingface"

# Accelerator to handle device setup
accelerator = Accelerator()
device = accelerator.device

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=token,
    cache_dir=cache_dir,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    token=token,
    cache_dir=cache_dir,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa"
)

# Set pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.config.pad_token_id = tokenizer.pad_token_id
