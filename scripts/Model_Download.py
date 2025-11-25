from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
    local_dir="models/mistral-7b-gptq",
    local_dir_use_symlinks=False
)

print("Model downloaded successfully.")
