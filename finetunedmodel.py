from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("./finetuned-mistral", torch_dtype=torch.float32)
model.to("cpu")
tokenizer = AutoTokenizer.from_pretrained("./finetuned-mistral")

def ask(question):
    prompt = f"<s>[INST] {question.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    return answer.split("[/INST]")[-1].strip()
