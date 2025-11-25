from datasets import load_dataset

def download_dolly():
    print("\n📥 Downloading Dolly-15k (High-Quality Instructions)...")
    ds = load_dataset("databricks/databricks-dolly-15k")
    ds.save_to_disk("data/dolly-15k")
    print("✅ Saved to data/dolly-15k")

def download_ultrachat():
    print("\n📥 Downloading UltraChat-200k (Conversational Dataset)...")
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    ds.save_to_disk("data/ultrachat-200k")
    print("✅ Saved to data/ultrachat-200k")

def download_codealpaca():
    print("\n📥 Downloading CodeAlpaca-20k (Coding Instructions)...")
    ds = load_dataset("yahma/alpaca-cleaned")
    ds.save_to_disk("data/codealpaca-20k")
    print("✅ Saved to data/codealpaca-20k")

if __name__ == "__main__":
    print("🚀 Starting dataset download...")
    
    download_dolly()
    download_ultrachat()
    download_codealpaca()

    print("\n🎉 ALL DATASETS DOWNLOADED SUCCESSFULLY!")
