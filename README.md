# SEXUAL-HEALTH-AI-ASSISTANT
The Sexual Health AI Assistant uses advanced AI to provide empathetic, professional advice on sexual health questions. Fine-tuned for precise, supportive responses, this project ensures privacy and accessibility with a user-friendly Streamlit interface, offering reliable guidance on sensitive topics to promote well-being. 

# 🧠 LLM for Sexual Health Guidance (All Genders)

A fine-tuned conversational AI model built using TinyLlama, designed to provide empathetic, informative, and judgment-free sexual health advice for people of all genders and age groups. This project aims to break the stigma around sexual health concerns by offering accessible, private, and medically-aware answers.

---

## 🌟 Key Features

- 🔬 Fine-tuned on a curated sexual health Q&A dataset
- 🗣️ Real-time natural language interaction (CLI + Streamlit app)
- 🧠 Doctor-like tone with empathy and factual responses
- 🔄 LoRA-based lightweight fine-tuning (PEFT)
- 👥 Inclusive of all age ranges and genders

---

## 📂 Project Structure
llm-sexual-health-ai/
├── data/
│ ├── combined.csv # Raw Q&A dataset
│ └── tinyllama_train.jsonl # Converted instruction-based data
├── models/
│ └── finetuned-tinyllama/ # Saved model and tokenizer
├── app.py # CLI with system prompts
├── inference.py # Interactive Q&A with age input
├── finetuning.py # PEFT LoRA-based training script
├── finetunedmodel.py # Alternate model inference (Mistral)
├── run_app.py # Launch script for Streamlit
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── LICENSE # MIT License

---

## 🧠 Model & Training

- **Base Model:** [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- **Fine-Tuning Framework:** 🤗 Transformers + PEFT + Datasets
- **Technique:** LoRA (Low-Rank Adaptation)
- **Token Limit:** 512 max tokens
- **Sampling Parameters:**  
  - `temperature`: 0.5–0.7  
  - `top_k`: 35–50  
  - `top_p`: 0.8–0.9

---

##🔄 Finetuning Pipeline (LoRA)
Load and clean data from combined.csv

Convert to instruction format → tinyllama_train.jsonl

Apply tokenizer + max tokenization

Configure PEFT LoRA

Train with HuggingFace Trainer API

## 💻 How to Run

### 🔧 Installation

```bash
git clone https://github.com/YOUR_USERNAME/llm-sexual-health-ai.git
cd llm-sexual-health-ai
pip install -r requirements.txt 

##🚀 Launch CLI Chatbot:
python app.py

##🖥️ Launch Streamlit Web Interface
python run_app.py

Save fine-tuned model to models/finetuned-tinyllama/

##🛡️ Ethical Considerations

No real user data was used.

Advice is general and not a replacement for professional diagnosis.

Focused on promoting safe, respectful, and inclusive conversations.

Responses were carefully validated for tone, clarity, and bias reduction.
