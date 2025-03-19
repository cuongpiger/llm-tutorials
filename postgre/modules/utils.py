from typing import Dict, Any, Optional, List
from langchain.schema import Document

def load_env_to_dict(file_path: str) -> Optional[Dict[str, Any]]:
    env_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Remove whitespace and ignore comments or empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split the line into key and value
            key, value = line.split('=', 1)
            env_dict[key.strip()] = value.strip()
    return env_dict

def print_documents(documents: List[Document]):
    print("Amount of documents:", len(documents))
    for doc_th, document in enumerate(documents):
        print(f"Document {doc_th}:\n{document.page_content}", end="\n\n"+ 100*"-")

def from_documents_to_context(documents: List[Document]) -> str:
    context = ""
    for doc_th, document in enumerate(documents):
        context = context + f"- Document {doc_th + 1}:\n{document.page_content}\n\n"

    return context