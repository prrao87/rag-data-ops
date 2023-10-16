# RAG Data Ops

Example code to handle data ops for RAG applications via LangChain and LlamaIndex.

* Handling of multiple files for retrieval
* Chunking text to deal with context length issues of embedding models
* Processing files and data structures for multi-modal retrieval

## Setup

Install the dependencies in virtual environment via requirements.txt.

```sh
# Setup the environment for the first time
python -m venv .venv  # python -> python 3.10+

# Activate the environment (for subsequent runs)
source .venv/bin/activate

python -m pip install -r requirements.txt
```