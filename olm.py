import requests
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
"""""

# URL of the PDF you want to download
url = "https://rnesm.justice.gov.ma/Documentation/MA/4_ONC_Law_fr-FR.pdf"  # Replace with the actual URL of your PD
# this is the part when we talk about url exctract and reading it binary file without stocking it in desk
response = requests.get(url)
pdf_file = BytesIO(response.content)

# Load the PDF using LangChain
pdf_loader = PyPDFLoader(pdf_file)
documents = pdf_loader.load()
"""""
#download the fil
pdf_file_path = "C:/Users/hp/Downloads/law.pdf" # Replace with the actual file path
pdf_loader = PyPDFLoader(pdf_file_path)
documents = pdf_loader.load()
print(documents)

#clean text
import re
def clean_pdf_text(raw_documents):
    """
    Cleans and normalizes the extracted text from the PDF.
    """
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


# Clean the text
cleaned_text = clean_pdf_text(documents)
print("Cleaned text preview:")
print(cleaned_text[:1000])  # Preview the first 1000 characters of the cleaned text

#split
def split_by_articles_with_finditer(text):
    """
    Splits the cleaned text into chunks based on "Article X" patterns using re.finditer.
    """
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

# Split the cleaned text into articles
article_chunks = split_by_articles_with_finditer(cleaned_text)

# Display the first 5 articles for verification
print("\n--- Article Chunks Preview ---\n")
for article in article_chunks[:5]:
    print(article)
    print("\n---\n")
#embidding etape
  #vector storage
# load embeddings
# embedding = GPT4AllEmbeddings()
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from uuid import uuid4
import os
# Check if the vector store already exists

# Define the path to the persisted vector store
persist_directory = "./chroma_langchain_db"
embeddings = OllamaEmbeddings(model="nomic-embed-text")
if os.path.exists(persist_directory):
    print("Loading precomputed vector store...")
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

else:
    print("Vector store not found. Computing embeddings...")
    vector_store = Chroma(
        collection_name="legal_contract_collection",  # A name for your collection
        embedding_function=embeddings,  # The embedding model (OllamaEmbeddings in your case)
        persist_directory="./chroma_langchain_db",  # Path to persist/store the embeddings locally (optional)
    )
    uuids = [str(uuid4()) for _ in range(len(article_chunks))]
    documents = [Document(page_content=chunk, metadata={"source": "legal_articles"}) for chunk in article_chunks]
    vector_store.add_documents(documents=documents, ids=uuids)
    vector_store.persist()
#retreive
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

# Initialize model and retriever
retriever = vector_store.as_retriever()
model = ChatOllama(model="llama-2-7b-chat")  # Replace with your preferred remote model

qa_chain = RetrievalQA.from_chain_type(
    llm=model,
    retriever=retriever,
    return_source_documents=True  # To trace which legal reference was used
)


