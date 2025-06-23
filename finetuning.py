import pandas as pd
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType

# Load and preprocess CSV
df = pd.read_csv(r"D:\sexual_health_ai_project\qa_data\combined.csv", on_bad_lines='skip')

with open("tinyllama_train.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = row["question"]
        completion = row["answer"]
        age = row.get("age_range", "unknown age")
        data = {
            "prompt": f"<s>[INST] Age: {age}. Question: {prompt.strip()} [/INST]",
            "completion": f"{completion.strip()}</s>"
        }
        f.write(json.dumps(data) + "\n")

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("cpu")

# Load dataset
dataset = load_dataset("json", data_files="tinyllama_train.jsonl")["train"]

# Tokenization function
def tokenize(example):
    return tokenizer(
        example["prompt"] + example["completion"],
        truncation=True,
        padding="max_length",
        max_length=512
    )

tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

# LoRA configuration adjustments
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    optimizer_type="adamw_torch",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./finetuned-tinyllama-sexual-health",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    num_train_epochs=12,  # Increase for improved learning
    learning_rate=1e-4,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./finetuned-tinyllama")
tokenizer.save_pretrained("./finetuned-tinyllama")
