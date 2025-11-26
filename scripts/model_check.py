import os

model_path = "models/mistral-7b-instruct-v0.2-fp16"
print("Files:", len(os.listdir(model_path)))
print("Total size (GB):", sum(os.path.getsize(os.path.join(model_path, f)) for f in os.listdir(model_path)) / 1e9)
