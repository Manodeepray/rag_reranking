from langchain_huggingface import HuggingFaceEmbeddings
import getpass
import os
from artifacts import keys
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key # keys.NVIDIA_EMBED_API_KEY
    
small_embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
large_embedding_model = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")