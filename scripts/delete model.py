import shutil

shutil.rmtree("models/mistral-7b-instruct-v0.2-fp16", ignore_errors=True)
print("Deleted model folder.")
