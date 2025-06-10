import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gradio as gr


MODEL_ENV_VARS = ["MODEL_PATH", "MODEL_NAME"]


def load_model():
    """Load the language model specified by environment variables.

    ``MODEL_PATH`` takes precedence over ``MODEL_NAME``. Both can point to a local
    directory containing the model files or a model ID from Hugging Face.
    """

    model_source = None
    for var in MODEL_ENV_VARS:
        if var in os.environ:
            model_source = os.environ[var]
            break
    if model_source is None:
        model_source = "openlm-research/open_llama_3b"

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
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
