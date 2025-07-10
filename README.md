# Git Repo RAG with LangChain and OpenAI

This project demonstrates how to use LangChain and OpenAI to build a **Retrieval-Augmented Generation (RAG)** pipeline that indexes code from GitHub repositories and enables question-answering over the codebase with line-level source information.

---

## Features

* Clone and parse GitHub repositories
* Load and chunk source code files
* Embed code using OpenAI embeddings
* Persist and retrieve with Chroma vectorstore
* Generate structured answers using LangChain’s Pydantic output parser
* Track code line numbers and modifications
* Interactive CLI-based QA interface
* Debug with raw and parsed outputs

---

## File Support

Supports loading files with the following extensions:

```txt
.txt, .md, .java, .json
```

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
```

### 2. Create and activate a virtual environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
---

## Configuration

### Repositories to Analyze

Modify the `REPO_URLS` list in the script to include your desired GitHub repositories:

```python
REPO_URLS = [
    "https://github.com/yourname/yourrepo.git",
    "https://github.com/another/repo.git"
]
```

### OpenAI API Key

Ensure the `OPENAI_API_KEY` is set in your environment:

```bash
export OPENAI_API_KEY=your-key-here
```

---

## How It Works

1. **Repository Cloning**: Clones each repo locally if not already present.
2. **Document Loading**: Loads files matching allowed extensions.
3. **Chunking**: Splits files into overlapping chunks using `RecursiveCharacterTextSplitter`.
4. **Embedding & Vector Store**: Embeds chunks using OpenAI embeddings and stores them in a Chroma vectorstore.
5. **RAG Pipeline**:

   * Retrieves relevant chunks for the question.
   * Uses `ChatOpenAI` with a detailed prompt and Pydantic output model.
   * Parses and optionally corrects the output using `OutputFixingParser`.
6. **QA Output**: Returns a structured JSON output showing:

   * Source file paths
   * Line numbers and content
   * Suggested code modifications (if applicable)

---

## Example Usage

Run the script:

```bash
python your_script.py
```

Then enter your questions:

```
Ask a question (or type 'exit'): what does the main method do in whiteboarding repo?

Question: ...
Raw LLM Answer (for debugging): ...
Parsed Output (Actor object): ...
Source Documents:
  - Source: whiteboarding/main.py
...
```

---

## Example Output

```json
{
  "source_files": ["main.py"],
  "source": ["def main():", "  print('Hello')"],
  "source_line_numbers": "10:12",
  "source_line": ["def main():", "  print('Hello')"],
  "modified_code": ["def main():", "  print('Hello, world!')"],
  "modified_code_for_pr": ["def main():", "  print('Hello, world!')"]
}
```

---

## Project Structure

```
.
├── main.py                 # Main RAG pipeline
├── chroma_db/             # Persisted vectorstore
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
```

---

## Notes

* Supports LangChain Expression Language (LCEL) and the latest `Runnable` chains.
* Output fixing ensures compliance with Pydantic schema even if LLM responses deviate.
* Built-in debug logs help trace errors in parsing or retrieval.

---

## Powered by

* [LangChain](https://www.langchain.com/)
* [OpenAI](https://platform.openai.com/)
* [Chroma](https://www.trychroma.com/)
* [Pydantic](https://docs.pydantic.dev/)

---

## License

MIT License. See `LICENSE` file.

---
