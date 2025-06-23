from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_path = "./finetuned-tinyllama"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cpu")

# Define system prompt template
system_prompt_template = """Your job is to:
- Understand the user's question or concern clearly.
- Respond like a medically aware professional doctor would.
- Provide clear, empathetic answers and practical suggestions.

Use the provided context below if it's useful. If the context is unrelated or missing info, you may use your own medical knowledge to help the user.

{context}

Answer in a clear, structured, empathetic tone."""

context = "Recent checkup showed no irregularities."

while True:
    question = input("Ask a sexual health question (or type 'exit'): ")
    if question.lower() == 'exit':
        break

    # Fill in the system prompt
    system_prompt = system_prompt_template.format(context=context)

    # Craft prompt and ensure it fits the intended purpose
    prompt = f"{system_prompt}\n\n[s] Question: {question.strip()} [/s]"
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

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Find the start of the actual answer
    try:
        answer = output_text.split("Question:")[1].strip()
    except IndexError:
        answer = output_text.strip()
    print("\n🩺 AI Doctor:\n", answer, "\n")
