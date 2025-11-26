import torch
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model


# -----------------------------------------------------------
# 1. Model Configuration (adjust model name here)
# -----------------------------------------------------------
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"   # or your local model path

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)


# -----------------------------------------------------------
# 2. Load Tokenizer and Base Model
# -----------------------------------------------------------
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
)


# -----------------------------------------------------------
# 3. Load the 3 Datasets
# -----------------------------------------------------------
print("Loading datasets...")

dolly = load_dataset("databricks/databricks-dolly-15k")["train"]
ultra = load_dataset("HuggingFaceH4/ultrachat_200k")["train"]
code = load_dataset("yahma/alpaca-cleaned")["train"]   # CodeAlpaca compatible


# -----------------------------------------------------------
# 4. Convert All Datasets to a Single Unified Format
# -----------------------------------------------------------
def format_example(example):
    instruction = example.get("instruction") or example.get("context") or "You are an AI assistant."
    input_text = example.get("input", "")
    output_text = example.get("response") or example.get("output") or ""

    if input_text:
        final_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        final_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

    return {"text": final_text}


print("Formatting datasets...")

dolly = dolly.map(format_example)
ultra = ultra.map(format_example)
code = code.map(format_example)

combined_dataset = concatenate_datasets([dolly, ultra, code])


# -----------------------------------------------------------
# 5. Tokenization Function
# -----------------------------------------------------------
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=512,
    )


print("Tokenizing...")
tokenized_dataset = combined_dataset.map(tokenize, batched=True)


# -----------------------------------------------------------
# 6. LoRA Fine-Tuning Parameters
# -----------------------------------------------------------
lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)


# -----------------------------------------------------------
# 7. Training Arguments
# -----------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./model-out",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    logging_steps=25,
    save_steps=500,
    learning_rate=2e-4,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none",
)


# -----------------------------------------------------------
# 8. Trainer
# -----------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)


# -----------------------------------------------------------
# 9. Train & Save
# -----------------------------------------------------------
print("Starting training...")
trainer.train()

print("Saving model...")
model.save_pretrained("./model-out")

print("Training complete.")
