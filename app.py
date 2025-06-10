import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


def load_model():
    model_name = os.getenv("MODEL_NAME", "openlm-research/open_llama_3b")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model, tokenizer

model, tokenizer = load_model()


def generate_response(message, history):
    conversation = ""
    for user_msg, bot_msg in history:
        conversation += f"<s>[INST] {user_msg} [/INST] {bot_msg}</s> "
    conversation += f"<s>[INST] {message} [/INST]"
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, top_p=0.9)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # only return newly generated text after last prompt
    response = response.split("[/INST]")[-1].strip()
    history.append((message, response))
    return response

iface = gr.ChatInterface(fn=generate_response, title="LLaMA Chat", description="A simple web interface for chatting with a LLaMA model.")

if __name__ == "__main__":
    iface.launch()
