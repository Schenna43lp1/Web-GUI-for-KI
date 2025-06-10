# Web-GUI-for-KI

This project provides a minimal web user interface for interacting with a LLaMA-based language model.

## Installation

Install the dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with:

```bash
python app.py
```

By default the script loads the `openlm-research/open_llama_3b` model from Hugging Face. You can specify a different model by setting the `MODEL_NAME` environment variable:

```bash
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf python app.py
```

The web interface will start and provide a simple chat view similar to ChatGPT.

## Notes

Running large models such as Llama 2 may require a GPU with sufficient memory. You can start with a smaller model if you encounter resource issues.
