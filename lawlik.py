import requests
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from uuid import uuid4
import os
import tempfile
#download the ref file containing contracts law
pdf_file_path = "C:/Users/hp/Downloads/law.pdf" # Replace with the actual file path
pdf_loader = PyPDFLoader(pdf_file_path)
documents = pdf_loader.load()


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


#code for the front end
import streamlit as st
from PIL import Image
import base64
from io import BytesIO

# Configuration de la page pour un affichage large
st.set_page_config(layout="wide")

# CSS personnalisé pour l'interface
st.markdown("""
    <style>
        h1 {
            font-size: 1.5em;
        }
        h3 {
            font-size: 1.2em;
        }
        p, button {
            font-size: 1em;
        }

        .stButton>button {
            width: 100%;
            font-size: 14px;
            padding: 12px 0;
        }

        /* Style du header */
        .custom-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 10px 20px;
            background-color: #f8f9fa;
            border-bottom: 2px solid #dee2e6;
            box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.1);
        }

        .custom-header .menu a {
            text-decoration: none;
            color: #007bff;
            font-weight: 500;
        }
        .custom-header .menu a:hover {
            color: #0056b3;
            text-decoration: underline;
        }

        /* Media Query pour les petits écrans */
        @media only screen and (max-width: 600px) {
            h1 {
                font-size: 1.5em !important;
            }
            h3 {
                font-size: 1.2em !important;
            }
            p, button {
                font-size: 1em !important;
            }

            .stButton>button {
                font-size: 12px !important;
                padding: 10px 0;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Fonction utilitaire pour convertir une image en base64
def get_image_base64(image):
    """
    Convertit une image PIL en chaîne base64 pour utilisation dans HTML/CSS.
    """
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_string = base64.b64encode(buffered.getvalue()).decode()
    return encoded_string

# Fonction pour afficher le header avec un logo local redimensionné
def display_header():
    # Chargez l'image du logo
    logo_path = "C:/Users/hp/Downloads/logo.jpeg"  # Remplacez par votre chemin
    logo = Image.open(logo_path)

    # Affichez le header avec un logo redimensionné (1 cm ≈ 38px)
    st.markdown(f"""
        <div class="custom-header">
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{get_image_base64(logo)}" 
                     style="height: 30px; margin-right: 15px;" alt="Logo">
                <span style="font-size: 1.8em; font-weight: bold; color: #333;">LAWlik</span>
            </div>
            <div class="menu">
                <a href="#accueil" style="margin-right: 15px;">Accueil</a>
                <a href="#services">Services</a>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Fonction pour afficher la photo principale (avec ajustement de taille)
def display_main_photo():
    st.title("Mon espace juridique:")

    # Création des colonnes
    col1, col2 = st.columns([1, 2])  # 1/3 pour col1, 2/3 pour col2

    # Image dans la colonne de gauche
    with col1:
        photo_path = "C:/Users/hp/Downloads/carte.jpeg"  
        photo = Image.open(photo_path)
        st.image(photo, use_container_width=True, caption="Votre carte juridique :")

    # Texte explicatif dans la colonne de droite
    with col2:
        st.markdown("""Bienvenue dans votre espace juridique, où innovation et protection se rencontrent ! \n
🌟 Devenez maître de vos droits : posez vos questions à notre assistant virtuel intelligent et obtenez des réponses instantanées.\n
🔍 Analysez vos contrats : expliquez les clauses et repérez les zones à risque en un clic.\n
🛡️ Prédisez l'avenir : prévoyer vos problèmes juridiques avant qu'ils ne surviennent.\n
🎖️ Votre Carte juridique : Plus vous explorez et interagissez avec l’application, plus vous gagnez d’XP, que vous pouvez ensuite utiliser pour déverrouiller des analyses de contrats détaillées.
\n
🔓 Un espace unique pour vous accompagner dans chaque étape de votre vie juridique.   
                         """)

# Interface principale (Page 2 - Espace juridique)
def show_legal_interface():
    display_header()  # Affiche le header
    display_main_photo()  # Affiche l'image principale

    # Ajouter la fonctionnalité de téléchargement de fichier
    st.markdown("### Uploadez vos contrats pour analyse :")
    uploaded_file = st.file_uploader("Choisissez un fichier (PDF, Word)", type=["pdf", "docx"])
    
    if uploaded_file is not None:  # Ensure the user uploaded a file
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load the PDF using the file path
        pdf_loader = PyPDFLoader(temp_file_path)
        
        # Process the PDF file
        user_documents = pdf_loader.load()
        # Continue processing `documents`

        # Clean and split the uploaded document into clauses
        cleaned_contract = clean_pdf_text(user_documents)
        contract_chunks = split_by_articles_with_finditer(cleaned_contract)

        # Perform clause analysis
        st.markdown("### Analyse des clauses :")
        results = analyze_contract_clauses(contract_chunks, qa_chain)
        for result in results:
            st.markdown(f"**Clause :**\n{result['clause']}")
            st.markdown(f"**Analyse :**\n{result['analysis']}")
            if result["sources"]:
                st.markdown("**Sources utilisées pour l'analyse :**")
                for source in result["sources"]:
                    st.markdown(f"- {source['metadata']['source']}")
            else:
                st.markdown("*Aucune source trouvée pour cette clause.*")
            st.markdown("---")

    # Perform compliance comparison with reference law
    st.markdown("### Comparaison avec la loi de référence :")
    anomalies = compare_with_reference(contract_chunks, qa_chain, retriever)
    for anomaly in anomalies:
        st.markdown(f"**Clause :**\n{anomaly['clause']}")
        st.markdown(f"**Problème identifié :** {anomaly['flag_reason']}")
        st.markdown(f"**Détails :**\n{anomaly['details']}")
        if anomaly["sources"]:
            st.markdown("**Articles de référence potentiellement en conflit :**")
            for source in anomaly["sources"]:
                st.markdown(f"- {source['source']}")
        else:
            st.markdown("*Aucun article de référence trouvé pour cette clause.*")
        st.markdown("---")


    st.markdown("### Vous avez sélectionné les attributs suivants :")
    st.write(", ".join(st.session_state.selected_professions))
    st.markdown("COMING SOON : conseils, prévention et explications de vos droits")

    if st.button("Retour"):
        st.session_state.show_new_interface = False
        st.session_state.selected_professions = []

# Liste des professions
professions = [
    "Étudiant", "Salarié", "Auto-entrepreneur", 
    "Propriétaire", "Locataire", "Commerçant", 
    "Retraité", "Étranger au Maroc", 
    "Handicap", "Conducteur", 
    "Association", "Entreprise"
]

# Gestion des états pour les transitions
if "selected_professions" not in st.session_state:
    st.session_state.selected_professions = []
if "show_new_interface" not in st.session_state:
    st.session_state.show_new_interface = False

# Fonction pour ajouter/retirer des professions
def toggle_selection(profession):
    if profession in st.session_state.selected_professions:
        st.session_state.selected_professions.remove(profession)
    else:
        st.session_state.selected_professions.append(profession)

# Fonction pour soumettre la sélection
def submit_selection():
    if not st.session_state.selected_professions:
        st.warning("Veuillez sélectionner au moins un attribut avant de continuer.")
    else:
        st.session_state.show_new_interface = True

# Interface de sélection des professions
def show_profession_selection():
    st.title("Choisissez les attributs qui vous correspondent :")
    st.markdown("### Cliquez sur les boutons pour sélectionner:")

    # Affichage des professions avec boutons
    cols = st.columns(4)
    for idx, profession in enumerate(professions):
        with cols[idx % 4]:
            is_selected = profession in st.session_state.selected_professions
            button_label = f"⚖️ {profession}" if is_selected else profession
            if st.button(button_label, key=profession):
                toggle_selection(profession)

    st.markdown("### Une fois terminé, cliquez sur le bouton 'Soumettre'.")
    st.button("Soumettre", on_click=submit_selection)

# Logique principale
if not st.session_state.show_new_interface:
    show_profession_selection()  # Affiche la sélection des professions
else:
    show_legal_interface()  # Affiche l'interface principale