from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir="models/mistral-7b-instruct-v0.2-fp16",
    local_dir_use_symlinks=False,
    revision="main"
)

print("Full 35GB FP16 model downloaded successfully.")
