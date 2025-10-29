import requests
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from uuid import uuid4
import os
#download the ref file containing contracts law
pdf_file_path = "C:/Users/hp/Downloads/law.pdf" # Replace with the actual file path
pdf_loader = PyPDFLoader(pdf_file_path)
documents = pdf_loader.load()
# download the contrat by the user
user_file_path = input("Please provide the path to your contract PDF file: ")
pdf_loader = PyPDFLoader(user_file_path)
user_documents = pdf_loader.load()
#functions for text
#clean text
import re
def clean_pdf_text(raw_documents):
    cleaned_text = []
    for doc in raw_documents:
        text = doc.page_content
        
        # Remove page numbers and patterns like "- 1 -"
        text = re.sub(r'-\s?\d+\s?-', '', text)
        
        # Normalize excessive whitespace and line breaks
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\sàâéèêôûç,;.!?]', '', text)
        
        # Trim leading and trailing spaces
        text = text.strip()
        
        cleaned_text.append(text)
    
    return " ".join(cleaned_text)

#split
def split_by_articles_with_finditer(text):
    # Regex to match "Article X" and capture its title and content
    pattern = r'(Article\s+\d+(-\d+)?)(.*?)(?=Article\s+\d+(-\d+)?|$)'
    
    # Find all matches for articles and their content
    matches = re.finditer(pattern, text, re.DOTALL)
    
    # Combine the title and content into chunks
    chunks = []
    for match in matches:
        title = match.group(1).strip()  # "Article X" (Title)
        content = match.group(3).strip()  # The content after the title
        chunks.append(f"{title}\n{content}")
    
    return chunks

# usage of functions

#for reference law
cleaned_reference = clean_pdf_text(documents)
article_chunks = split_by_articles_with_finditer(cleaned_reference)
#for contract
cleaned_contrat = clean_pdf_text(user_documents)
contrat_chunks = split_by_articles_with_finditer(cleaned_contrat)

#embedding and vector storage for the reference document
# load embeddings
# Define the path to the persisted vector store
persist_directory = "./chroma_langchain_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists(persist_directory):
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

else:
    vector_store = Chroma(
        collection_name="legal_contract_collection",  # A name for your collection
        embedding_function=embeddings,  # The embedding model (OllamaEmbeddings in your case)
        persist_directory="./chroma_langchain_db",  # Path to persist/store the embeddings locally (optional)
    )
    uuids = [str(uuid4()) for _ in range(len(article_chunks))]
    documents = [Document(page_content=chunk, metadata={"source": "legal_articles"}) for chunk in article_chunks]
    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.persist()
#embedding and vector storage for the reference document
# load embeddings
# Define the path to the persisted vector store
law_persist_directory = "./chroma_law_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Vérification et initialisation
if os.path.exists(law_persist_directory):
    law_vector_store = Chroma(persist_directory=law_persist_directory, embedding_function=embeddings)
else:
    law_vector_store = Chroma(
        collection_name="law_contract_collection",
        embedding_function=embeddings,
        persist_directory=law_persist_directory,
    )
    # Ajouter les documents
    law_documents = [Document(page_content=chunk, metadata={"source": "law_file"}) for chunk in article_chunks]
    uuids = [str(uuid4()) for _ in range(len(article_chunks))]
    law_vector_store.add_documents(documents=law_documents, ids=uuids)

#retreive
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# Initialize model and retriever
retriever = vector_store.as_retriever()
model = ChatOllama(model="llama3-chatqa")

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True  # To trace which legal reference was used
)
#analyser le contrat

def analyze_contract_clauses(clauses, qa_chain):
    results = []
    for clause in clauses:
        if clause.strip():
            try:
                # Use `invoke` to get all outputs
                response = qa_chain.invoke(f"Analyze this clause: {clause}")
                results.append({
                    "clause": clause,
                    "analysis": response.get("result", "No result returned"),
                    "sources": response.get("source_documents", [])
                })
            except Exception as e:
                print(f"Error analyzing clause: {e}")
                results.append({
                    "clause": clause,
                    "analysis": "Error during analysis",
                    "sources": []
                })
    return results


# Test the function with the user's contract chunks
test_results = analyze_contract_clauses(contrat_chunks, qa_chain)
# Print the analysis results
for result in test_results:
    print("Clause:")
    print(result["clause"])
    print("\nAnalysis:")
    print(result["analysis"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- Source: {source.metadata.get('source')}")
    print("\n" + "-" * 80 + "\n")


def compare_with_reference(contract_chunks, qa_chain, retriever):
    anomalies = []
    
    for clause in contract_chunks:
        if clause.strip():
            try:
                # Retrieve relevant law articles using similarity search
                retrieved_docs = retriever.get_relevant_documents(clause)  # Fix retrieval
                
                # Prepare the question for the QA model
                references = (
                    retrieved_docs[0].page_content if retrieved_docs else "No reference found"
                )
                question = f"Does this clause comply with the following legal references? {clause}\nReferences: {references}"
                
                # Analyze the clause using the QA model
                response = qa_chain.invoke(question)  # Returns a dictionary
                
                # Handle the response and check for anomalies
                if "violation" in response.get("result", "").lower() or "non-compliance" in response.get("result", "").lower():
                    anomalies.append({
                        "clause": clause,
                        "flag_reason": "Potential non-compliance with legal reference",
                        "details": response["result"],
                        "sources": [doc.metadata for doc in retrieved_docs] if retrieved_docs else []
                    })
            except Exception as e:
                print(f"Error analyzing clause: {e}")
                anomalies.append({
                    "clause": clause,
                    "flag_reason": "Error during comparison",
                    "details": str(e),
                    "sources": []
                })
    print("QA chain response:", response)
    print("Retrieved documents:", retrieved_docs)
    return anomalies

# Run anomaly detection
anomalies = compare_with_reference(contrat_chunks, qa_chain, retriever)

# Print anomalies
for anomaly in anomalies:
    print("Clause:")
    print(anomaly["clause"])
    print("\nFlag Reason:")
    print(anomaly["flag_reason"])
    print("\nDetails:")
    print(anomaly["details"])
    print("\nSources:")
    for source in anomaly["sources"]:
        print(f"- {source.get('source', 'Unknown')}")
    print("\n" + "-" * 80 + "\n")


