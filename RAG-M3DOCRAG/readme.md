# M3DOCRAG:Multi-modal Retrieval is What You Need for Multi-page Multi-document Understanding

A Python implementation of document understanding using vision-language models for retrieval and question answering across multiple documents.

## Installation

```bash
# Clone the repository
git clone https://github.com/ujjalkumarmaity/research-paper-implementation.git

cd RAG-M3DOCRAG

# Install dependencies
pip install -r requirements.txt

# Install system dependency for PDF processing
sudo apt-get install poppler-utils  # Ubuntu/Debian

```


## Command Line Usage

```bash
# Run with sample data
python ./src/m3docrag.py --dataset sample
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (2 GPUs recommended)
- 16GB+ GPU memory

## Project Structure

```
src/
├── m3docrag.py        # Main pipeline implementation
├── data.py            # Dataset loading and PDF processing
├── vector_store.py    # FAISS vector search
├── config.py          # Configuration settings
└── __init__.py
```

## Models Used

- **Retrieval**: [ColPali v1.2](https://huggingface.co/vidore/colpali-v1.2) - Vision-based document retrieval
- **QA**: [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4) - Visual question answering

## Refrences
- https://techcommunity.microsoft.com/blog/azure-ai-foundry-blog/introduction-to-ocr-free-vision-rag-using-colpali-for-complex-documents/4276357
- https://github.com/Omaralsaabi/M3DOCRAG/tree/main
