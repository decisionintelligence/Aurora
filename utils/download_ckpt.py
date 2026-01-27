import os
# Set domestic mirror source (hf-mirror.com)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download, login

login("your_hf_token")

# Define model repositories to download
models_to_download = [
    # ViT model and image processor (using google/vit-base-patch16-224-in21k)
    "google/vit-base-patch16-224-in21k",
    # BERT model and tokenizer (using google-bert/bert-base-uncased)
    "google-bert/bert-base-uncased"
]

# Download directory (models folder in current directory)
download_dir = "/home/Aurora/checkpoints"
os.makedirs(download_dir, exist_ok=True)  # Create directory if it doesn't exist

# Download each model in the list
for model_name in models_to_download:
    # Construct local save path, replacing slashes with underscores
    local_dir = os.path.join(download_dir, model_name.replace("/", "_"))

    print(f"Starting download of {model_name} to {local_dir}")

    # Download model files
    snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Save actual files instead of symlinks
        ignore_patterns=["*.bin.index.json"]  # Optional: ignore unnecessary files
    )

    print(f"Download completed for {model_name}\n")

print("All models downloaded successfully!")