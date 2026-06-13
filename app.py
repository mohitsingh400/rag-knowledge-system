import streamlit as st
import requests
import uuid
import numpy as np
import faiss
from dotenv import load_dotenv
import os
from sentence_transformers import SentenceTransformer

# ===============================
# Load environment variables
# ===============================
load_dotenv()

# =====================================================
# 🔑 PASTE YOUR GROQ API KEY HERE (get from console.groq.com)
# Option 1: Direct paste (for testing only)
# GROQ_API_KEY = "your_groq_api_key_here"
#
# Option 2: Use .env file (recommended)
# Create a .env file and add: GROQ_API_KEY=your_key_here
#
# Option 3: Streamlit Cloud → App Settings → Secrets
# Add: GROQ_API_KEY = 'your_key_here'
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # reads from .env or Streamlit secrets


# ===============================
# Initialize FAISS + Embedding Model
# ===============================
@st.cache_resource
def init_faiss():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    dimension = 384  # all-MiniLM-L6-v2 output size
    index = faiss.IndexFlatL2(dimension)
    return model, index

# In-memory document store
if "documents" not in st.session_state:
    st.session_state.documents = []  # list of {"id": ..., "text": ..., "title": ...}


# ===============================
# Groq API Call
# ===============================
def call_groq_api(prompt):
    """Call the Groq API with the given prompt"""
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",  # 🔑 API key used here
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama-3.3-70b-versatile",  # free Groq model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"


# ===============================
# Vector DB Functions (FAISS)
# ===============================
def add_document(model, index, text, title):
    """Embed and add document to FAISS index"""
    embedding = model.encode([text]).astype('float32')
    index.add(embedding)
    doc_id = str(uuid.uuid4())[:8]
    st.session_state.documents.append({"id": doc_id, "text": text, "title": title})
    return doc_id

def search_documents(model, index, query, n_results=3):
    """Search similar documents using FAISS"""
    if index.ntotal == 0:
        return []
    query_embedding = model.encode([query]).astype('float32')
    n_results = min(n_results, index.ntotal)
    distances, indices = index.search(query_embedding, n_results)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(st.session_state.documents):
            doc = st.session_state.documents[idx]
            results.append({
                "text": doc["text"],
                "title": doc["title"],
                "similarity": round(1 / (1 + dist), 3)
            })
    return results

def create_rag_prompt(query, context_docs):
    """Create RAG prompt combining query and retrieved context"""
    context = "\n\n".join([doc["text"] for doc in context_docs])
    return f"""Based on the following context information, please answer the question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""


# ===============================
# Streamlit UI
# ===============================
st.title("🤖 RAG Demo: Vector DB + LLM")
st.markdown("### Learn how Retrieval-Augmented Generation works!")

# Initialize FAISS
model, index = init_faiss()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📚 Add Knowledge", "🔍 Search Vector DB", "💬 Ask Questions", "🎓 Compare Responses"])

with tab1:
    st.header("Step 1: Build Your Knowledge Base")
    st.markdown("Add documents to the vector database. These will be used as context for RAG.")

    sample_docs = {
        "Machine Learning Basics": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning.",
        "Deep Learning": "Deep learning uses neural networks with multiple layers to learn representations of data. It is particularly effective for image recognition, speech recognition, and natural language processing tasks.",
        "Natural Language Processing": "NLP is a field of AI that focuses on the interaction between computers and human language. Key tasks include text classification, sentiment analysis, machine translation, and question answering.",
        "Computer Vision": "Computer vision enables machines to interpret and understand visual information from images and videos. Common applications include object detection, image segmentation, and facial recognition."
    }

    st.subheader("Quick Add: Sample Documents")
    for title, content in sample_docs.items():
        if st.button(f"Add: {title}"):
            doc_id = add_document(model, index, content, title)
            st.success(f"Added '{title}' to knowledge base! ID: {doc_id}")

    st.subheader("Add Custom Document")
    doc_title = st.text_input("Document Title")
    doc_content = st.text_area("Document Content", height=150)

    if st.button("Add Custom Document"):
        if doc_content and doc_title:
            doc_id = add_document(model, index, doc_content, doc_title)
            st.success(f"Added '{doc_title}' to knowledge base! ID: {doc_id}")
        else:
            st.error("Please provide both title and content")

with tab2:
    st.header("Step 2: Search the Vector Database")
    st.markdown("See how vector similarity search works - the core of RAG!")

    search_query = st.text_input("Search Query", placeholder="e.g., 'neural networks'")
    num_results = st.slider("Number of results", 1, 5, 3)

    if st.button("Search") and search_query:
        with st.spinner("Searching vector database..."):
            results = search_documents(model, index, search_query, num_results)

            if results:
                st.subheader("Search Results:")
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - Similarity: {result['similarity']}"):
                        st.write(f"**Title:** {result['title']}")
                        st.write(f"**Content:** {result['text']}")
            else:
                st.info("No documents found. Add some documents first!")

with tab3:
    st.header("Step 3: RAG in Action")
    st.markdown("Ask questions and see how RAG uses retrieved context to answer!")

    user_question = st.text_input("Your Question", placeholder="e.g., 'What is deep learning?'")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Answer with RAG"):
            if user_question:
                with st.spinner("Retrieving relevant documents..."):
                    results = search_documents(model, index, user_question, 2)

                    if results:
                        rag_prompt = create_rag_prompt(user_question, results)

                        st.subheader("🔍 Retrieved Context:")
                        for i, doc in enumerate(results):
                            st.text_area(f"Document {i+1}: {doc['title']}", doc['text'], height=100, disabled=True)

                        st.subheader("🤖 RAG Response:")
                        with st.spinner("Getting AI response..."):
                            response = call_groq_api(rag_prompt)
                            st.write(response)
                    else:
                        st.warning("No relevant documents found. Add documents first!")
            else:
                st.error("Please enter a question")

    with col2:
        if st.button("Answer without RAG"):
            if user_question:
                st.subheader("🤖 Direct LLM Response:")
                with st.spinner("Getting AI response..."):
                    response = call_groq_api(user_question)
                    st.write(response)
            else:
                st.error("Please enter a question")

with tab4:
    st.header("Step 4: Compare RAG vs Direct LLM")
    st.markdown("See the difference between RAG-enhanced and direct LLM responses!")

    comparison_question = st.text_input("Question for Comparison", placeholder="e.g., 'Explain computer vision'")

    if st.button("Compare Responses"):
        if comparison_question:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("🔗 RAG Response")
                with st.spinner("Generating RAG response..."):
                    results = search_documents(model, index, comparison_question, 2)
                    if results:
                        rag_prompt = create_rag_prompt(comparison_question, results)
                        rag_response = call_groq_api(rag_prompt)

                        st.write("**Context Used:**")
                        for i, doc in enumerate(results):
                            st.text_area(f"Context {i+1}", doc['text'][:200] + "...", height=80, disabled=True, key=f"rag_ctx_{i}")

                        st.write("**Response:**")
                        st.write(rag_response)
                    else:
                        st.warning("No context found. Add documents first!")

            with col2:
                st.subheader("🎯 Direct LLM Response")
                with st.spinner("Generating direct response..."):
                    direct_response = call_groq_api(comparison_question)
                    st.write("**Response:**")
                    st.write(direct_response)
        else:
            st.error("Please enter a question for comparison")

# ===============================
# Sidebar Info
# ===============================
st.sidebar.markdown(f"**Knowledge Base:** {len(st.session_state.documents)} documents")
st.sidebar.markdown(f"**FAISS Index:** {index.ntotal} vectors")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Llama 3.3-70b (Groq)")
st.sidebar.markdown("**Embeddings:** all-MiniLM-L6-v2")
st.sidebar.markdown("**Vector DB:** FAISS")
