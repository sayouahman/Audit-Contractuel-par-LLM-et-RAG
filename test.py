import os
import re
from io import BytesIO
from uuid import uuid4
from PIL import Image
import base64
import streamlit as st
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Initialisation des constantes
LAW_FILE_PATH = "C:/Users/hp/Downloads/law.pdf"  # À rendre configurable
PERSIST_DIR = "./chroma_langchain_db"
LAW_PERSIST_DIR = "./chroma_law_db"

# Charger les fichiers PDF de référence
def load_reference_documents(file_path):
    try:
        pdf_loader = PyPDFLoader(file_path)
        return pdf_loader.load()
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier de référence : {e}")
        return []

# Nettoyage de texte PDF
def clean_pdf_text(raw_documents):
    cleaned_text = []
    for doc in raw_documents:
        text = doc.page_content
        text = re.sub(r'-\s?\d+\s?-', '', text)  # Supprimer numéros de page
        text = re.sub(r'\s+', ' ', text)  # Normaliser espaces
        text = re.sub(r'[^\w\sàâéèêôûç,;.!?]', '', text)  # Supprimer caractères spéciaux
        text = text.strip()
        cleaned_text.append(text)
    return " ".join(cleaned_text)

# Fractionner par articles
def split_by_articles_with_finditer(text):
    pattern = r'(Article\s+\d+(-\d+)?)(.*?)(?=Article\s+\d+(-\d+)?|$)'
    matches = re.finditer(pattern, text, re.DOTALL)
    return [f"{match.group(1).strip()}\n{match.group(3).strip()}" for match in matches]

# Initialisation du stockage vectoriel
def init_vector_store(articles, persist_dir, collection_name="legal_contract_collection"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if os.path.exists(persist_dir):
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    else:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        uuids = [str(uuid4()) for _ in range(len(articles))]
        documents = [Document(page_content=chunk, metadata={"source": "legal_articles"}) for chunk in articles]
        vector_store.add_documents(documents=documents, ids=uuids)
        vector_store.persist()
        return vector_store

# Analyse des clauses
def analyze_contract_clauses(clauses, qa_chain):
    results = []
    for clause in clauses:
        if clause.strip():
            try:
                response = qa_chain.invoke(f"Analyze this clause: {clause}")
                results.append({
                    "clause": clause,
                    "analysis": response.get("result", "No result returned"),
                    "sources": response.get("source_documents", [])
                })
            except Exception as e:
                results.append({
                    "clause": clause,
                    "analysis": "Error during analysis",
                    "sources": []
                })
    return results

# Interface utilisateur
def display_interface():
    st.title("Analyseur de Contrats Juridiques")
    uploaded_file = st.file_uploader("Téléchargez votre contrat (PDF uniquement)", type=["pdf"])
    if uploaded_file:
        st.success("Fichier téléchargé avec succès.")
        pdf_loader = PyPDFLoader(BytesIO(uploaded_file.read()))
        user_documents = pdf_loader.load()
        cleaned_contract = clean_pdf_text(user_documents)
        contract_chunks = split_by_articles_with_finditer(cleaned_contract)

        reference_docs = load_reference_documents(LAW_FILE_PATH)
        cleaned_reference = clean_pdf_text(reference_docs)
        article_chunks = split_by_articles_with_finditer(cleaned_reference)
        
        vector_store = init_vector_store(article_chunks, PERSIST_DIR)
        retriever = vector_store.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOllama(model="llama3-chatqa"),
            retriever=retriever,
            return_source_documents=True,
        )
        
        # Analyse
        results = analyze_contract_clauses(contract_chunks, qa_chain)
        for result in results:
            st.write(f"Clause: {result['clause']}")
            st.write(f"Analyse: {result['analysis']}")
            st.write("---")

# Démarrer l'application
if __name__ == "__main__":
    display_interface()
