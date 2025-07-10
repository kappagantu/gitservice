from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from typing import List, Dict
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain.output_parsers import OutputFixingParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain.output_parsers import OutputFixingParser


import os
import subprocess

# --- Configuration ---
REPO_URLS = ["https://github.com/kappagantu/whiteboarding.git", "https://github.com/kappagantu/demo.git"]
repo_paths = []
ALLOWED_FILE_EXTENSIONS = ['.txt', '.md', '.java', '.json']
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
VECTORSTORE_PATH = "chroma_db"

# --- Utility function to extract repo name ---
def extract_repo_name_from_url(repo_url):
    last_slash_index = repo_url.rfind('/')
    if last_slash_index != -1:
        repo_with_extension = repo_url[last_slash_index + 1:]
        if repo_with_extension.endswith(".git"):
            return repo_with_extension[:-4]
    return None

# --- Checkout Repository ---
def checkout_repo(repo_url, destination_path):
    if not os.path.exists(destination_path):
        print(f"Cloning '{repo_url}' to '{destination_path}'...")
        try:
            subprocess.run(['git', 'clone', repo_url, destination_path], check=True)
            repo_paths.append(destination_path)
            print("Cloned successfully.")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error cloning: {e}")
            return False
    else:
        print(f"Repo exists at '{destination_path}'. Skipping clone.")
        repo_paths.append(destination_path)  # Add existing path
        return True

for REPO_URL in REPO_URLS:
    repo_name = extract_repo_name_from_url(REPO_URL)
    if not checkout_repo(REPO_URL, repo_name):
        print("Failed to ensure repository exists. Exiting.")
        exit()

# --- Load and Process Documents ---
def load_and_process_documents(repo_paths, allowed_extensions):
    documents = []
    for repo_path in repo_paths:
        loader = DirectoryLoader(repo_path, glob=[f"**/*{ext}" for ext in allowed_extensions], loader_cls=TextLoader)
        loaded_docs = loader.load()
        print('**********', repo_path, ' : ', [doc.metadata['source'] for doc in loaded_docs]) # Print source paths
        documents.extend(loaded_docs)
    return documents

documents = load_and_process_documents(repo_paths, ALLOWED_FILE_EXTENSIONS)
print(f"Loaded {len(documents)} documents.")

# --- Text Splitting ---
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

# --- Vector Store Creation ---
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=VECTORSTORE_PATH)
vectorstore.persist()
print(f"Created and persisted vector store at '{VECTORSTORE_PATH}'.")

# --- RAG Chain with Source Information ---
def create_rag_chain_with_source(llm, vectorstore):
    from langchain.prompts import PromptTemplate

def list_files(dir_to_use: str) -> List[str]:
    """List files in the current directory."""
    return os.listdir(dir_to_use)

def read_file(file_name: str) -> str:
    """Read a file's contents."""
    try:
        with open(file_name, "r") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: {file_name} not found."
    except Exception as e:
        return f"Error: {str(e)}"
        


class SourceModel(BaseModel):
    source_files: List[str] = Field(description="list of source file paths")
    source: List[str] = Field(description="source code lines")
    source_line_numbers: str = Field(description="provide start and end line number of the source code in the format start:end")
    source_line: List[str] = Field(description="source from that lines")
    modified_code: List[str] = Field(description="source code lines including fix")
    modified_code_for_pr: List[str] = Field(description="provide modified source code for pull request")

# --- RAG Chain with Source Information ---

def create_rag_chain_with_source(llm, vectorstore):
    # 1. Define the Pydantic output parser
    output_parser = PydanticOutputParser(pydantic_object=SourceModel)

    # 2. Get format instructions for the prompt
    format_instructions = output_parser.get_format_instructions()

    # 3. Define the prompt for the QA chain
    
    
    templatelatest = """You are a highly accurate AI assistant. Your sole task is to answer the user's question based STRICTLY on the provided context. 
    Instructions:
    1. Read the source file content line by line.
    2. Count ALL lines — including blank lines — as part of the source code. Do not skip any line.
    3. When referring to specific code segments, use the correct start and end line numbers based on the original context, counting from the first line.
    4. If the user requests a fix, rewrite the affected part of the code. When possible, use `if`/`else` instead of `try`/`catch`.
    5. Combine fixes logically to reduce redundancy.

    Response format:
    - Return a JSON object that conforms exactly to the Pydantic schema below.
    - Do NOT include any extra text, headers, explanations, or characters before or after the JSON block.
    - If there is no answer based on the context, return the following empty JSON structure exactly:
    ```json
    {{"source_files": [], "source_code_snippets": []}}
    Context:
    {context}

    Question: {question}

    {format_instructions}
    Answer (ONLY the JSON object):"""

    # Added "Answer (ONLY the JSON object):" for stronger instruction

    QA_CHAIN_PROMPT = PromptTemplate.from_template(templatelatest).partial(
        format_instructions=format_instructions
    )

    # 4. Create the core document chain: LLM + prompt to generate an answer from context
    document_chain = create_stuff_documents_chain(llm, QA_CHAIN_PROMPT)

    # 5. Create the retrieval chain: Retriever + document chain
    retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), document_chain)

    # 6. Create the OutputFixingParser instance
    fixed_output_parser = OutputFixingParser.from_llm(
        parser=output_parser,  # The original parser to try first
        llm=llm               # The LLM to use for fixing if parsing fails
    )

    # 7. Construct the final QA pipeline using LangChain Expression Language (LCEL)
        qa_chain = (
        RunnablePassthrough.assign(
            input=RunnableLambda(lambda x: x["question"])
        )
        | retrieval_chain # This returns {'answer': ..., 'context': ...}
        | RunnableParallel({
            "parsed_output": RunnableLambda(lambda x: x["answer"]) | fixed_output_parser,
            "source_documents": RunnableLambda(lambda x: x["context"]),
            "raw_answer": RunnableLambda(lambda x: x["answer"]) # ADD THIS LINE FOR DEBUGGING
        })
    )

    return qa_chain

# Instantiate LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0) # Use a recent chat model
# For even better results, consider "gpt-4o" or "gpt-4-turbo" if available
# Create the RAG chain
rag_chain = create_rag_chain_with_source(llm, vectorstore)

# --- Example Usage ---
if __name__ == "__main__":
    while True:
        query = input("Ask a question (or type 'exit'): ")
        if query.lower() == 'exit':
            break
        with get_openai_callback() as cb:
            try:
                # Invoke the rag_chain with the question
                # The input should be a dictionary with a 'question' key
                result = rag_chain.invoke({"question": query})

                print("\nQuestion:", query)
                # Print the raw answer from the LLM for debugging parsing issues
                print("Raw LLM Answer (for debugging):", result.get("raw_answer"))
                print("Parsed Output (Actor object):", result["parsed_output"])
                print("Type of Parsed Output:", type(result["parsed_output"]))

                print("\nSource Documents:")
                if result["source_documents"]:
                    for doc in result["source_documents"]:
                        print(f"  - Source: {doc.metadata.get('source', 'N/A')}")
                else:
                    print("  No source documents found.")

            except Exception as e:
                print(f"\nAn error occurred during chain invocation: {e}")
                print("Please check the raw LLM output and your Pydantic model definition.")

            print("-" * 50)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print("-" * 50)
