from huggingface_hub import snapshot_download

model_name = "deepseek-ai/DeepSeek-Coder-33B-instruct"   # << FIXED MODEL NAME (use this only)

# Downloads entire model to /models/deepseek_coder_33b

snapshot_download(
    repo_id=model_name,
    local_dir="./models/deepseek_coder_33b",
    local_dir_use_symlinks=False
)

print("Model downloaded successfully at ./models/deepseek_coder_33b")
