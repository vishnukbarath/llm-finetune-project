from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    revision="main",
    local_dir="models/mistral-7b-instruct-v0.2-fp16",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "*.safetensors",
        "config.json",
        "generation_config.json",
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
)

print("Full FP16 model downloaded!")
