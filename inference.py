from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./finetuned-tinyllama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cpu")

while True:
    # age = input("Enter your age range: ")
    question = input("Ask a sexual health question (or type 'exit'): ")
    if question.lower() == 'exit':
        break

    prompt = f"<s>[INST] Question: {question.strip()} [/INST]</s>"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.5,
            top_k=35,
            top_p=0.8
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n🩺 AI Doctor:\n", answer.split("[/INST]")[-1].strip(), "\n")
