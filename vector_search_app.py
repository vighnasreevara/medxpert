import streamlit as st
import os
import chromadb
from dotenv import load_dotenv
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="medxpert/chroma_db_fresh"  # 🔁 Change path if needed
))

# Load your collection
collection = chroma_client.get_collection(name="drug_images")  # 💡 Adjust if you used another name

# Load same embedding model used to build ChromaDB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to search ChromaDB
def search_chroma(query: str, top_k: int = 3):
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results

# ---------------- Streamlit UI ---------------- #

st.set_page_config(page_title="🔎 Chroma Vector Search", layout="centered")
st.title("🔍 MedXpert - Vector Search from Drug Images")
st.markdown("Search medicines based on drug image embeddings stored in ChromaDB.")

# Input field
user_input = st.text_input("💬 Enter a symptom, condition, or keyword:")

# Search and display
if user_input:
    with st.spinner("🔍 Searching vector database..."):
        results = search_chroma(user_input)

    if results and results['documents']:
        for i, (text, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0]), start=1):
            st.markdown(f"### 🧠 Result {i}")
            st.markdown(f"📄 **OCR Extracted Text:** {text}")
            st.markdown(f"🗂️ **ZIP File:** `{meta.get('zip_file')}`")
            st.markdown(f"🖼️ **Image File:** `{meta.get('image_name')}`")

            # Optional: Display image if accessible via a URL or path
            # If you have actual URL path to the image, replace this logic
            image_path = f"drug_images/{meta.get('zip_file')}/{meta.get('image_name')}"
            if os.path.exists(image_path):
                st.image(image_path, caption="Drug Image", use_column_width=True)
            else:
                st.info("🖼️ Image preview not available.")
    else:
        st.warning("No matching documents found.")
