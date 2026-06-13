import streamlit as st
import chromadb
import requests
import uuid
from dotenv import load_dotenv
import os

# ===============================
# Load environment variables
# ===============================
load_dotenv()

# =====================================================
# 🔑 PASTE YOUR GROQ API KEY HERE (get from console.groq.com)
# Option 1: Direct paste (for testing only, not recommended for production)
# GROQ_API_KEY = "your_groq_api_key_here"
#
# Option 2: Use .env file (recommended)
# Create a .env file in same folder and add: GROQ_API_KEY=your_key_here
# =====================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # reads from .env file


# ===============================
# Initialize ChromaDB client
# ===============================
@st.cache_resource
def init_chroma():
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="knowledge_base",
        metadata={"hnsw:space": "cosine"}
    )
    return collection


# ===============================
# Groq API Call (replaces SkillCaptain)
# ===============================
def call_groq_api(prompt):
    """Call the Groq API with the given prompt"""
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",  # API key goes here
        "Content-Type": "application/json"
    }

    body = {
        "model": "llama-3.3-70b-versatile",  # free model, fast
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
# Vector DB Functions
# ===============================
def add_document_to_vector_db(collection, text, metadata=None):
    """Add a document to the vector database"""
    doc_id = str(uuid.uuid4())
    collection.add(
        documents=[text],
        metadatas=[metadata or {}],
        ids=[doc_id]
    )
    return doc_id

def search_similar_documents(collection, query, n_results=3):
    """Search for similar documents in the vector database"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def create_rag_prompt(query, context_docs):
    """Create a RAG prompt combining query and retrieved context"""
    context = "\n\n".join([doc for doc in context_docs])
    rag_prompt = f"""Based on the following context information, please answer the question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

Question: {query}

Answer:"""
    return rag_prompt


# ===============================
# Streamlit UI
# ===============================
st.title("🤖 RAG Demo: Vector DB + LLM")
st.markdown("### Learn how Retrieval-Augmented Generation works!")

# Initialize ChromaDB
collection = init_chroma()

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
            doc_id = add_document_to_vector_db(
                collection,
                content,
                {"title": title, "source": "sample"}
            )
            st.success(f"Added '{title}' to knowledge base! ID: {doc_id[:8]}...")

    st.subheader("Add Custom Document")
    doc_title = st.text_input("Document Title")
    doc_content = st.text_area("Document Content", height=150)

    if st.button("Add Custom Document"):
        if doc_content and doc_title:
            doc_id = add_document_to_vector_db(
                collection,
                doc_content,
                {"title": doc_title, "source": "custom"}
            )
            st.success(f"Added '{doc_title}' to knowledge base! ID: {doc_id[:8]}...")
        else:
            st.error("Please provide both title and content")

with tab2:
    st.header("Step 2: Search the Vector Database")
    st.markdown("See how vector similarity search works - the core of RAG!")

    search_query = st.text_input("Search Query", placeholder="e.g., 'neural networks'")
    num_results = st.slider("Number of results", 1, 5, 3)

    if st.button("Search") and search_query:
        with st.spinner("Searching vector database..."):
            results = search_similar_documents(collection, search_query, num_results)

            if results['documents'][0]:
                st.subheader("Search Results:")
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    with st.expander(f"Result {i+1} - Similarity: {1-distance:.3f}"):
                        st.write(f"**Title:** {metadata.get('title', 'Untitled')}")
                        st.write(f"**Content:** {doc}")
                        st.write(f"**Distance:** {distance:.3f}")
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
                    search_results = search_similar_documents(collection, user_question, 2)

                    if search_results['documents'][0]:
                        retrieved_docs = search_results['documents'][0]
                        rag_prompt = create_rag_prompt(user_question, retrieved_docs)

                        st.subheader("🔍 Retrieved Context:")
                        for i, doc in enumerate(retrieved_docs):
                            st.text_area(f"Document {i+1}", doc, height=100, disabled=True)

                        st.subheader("🤖 RAG Response:")
                        with st.spinner("Getting AI response..."):
                            response = call_groq_api(rag_prompt)  # ✅ Groq API
                            st.write(response)
                    else:
                        st.warning("No relevant documents found in the knowledge base!")
            else:
                st.error("Please enter a question")

    with col2:
        if st.button("Answer without RAG"):
            if user_question:
                st.subheader("🤖 Direct LLM Response:")
                with st.spinner("Getting AI response..."):
                    response = call_groq_api(user_question)  # ✅ Groq API
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
                    search_results = search_similar_documents(collection, comparison_question, 2)
                    if search_results['documents'][0]:
                        retrieved_docs = search_results['documents'][0]
                        rag_prompt = create_rag_prompt(comparison_question, retrieved_docs)
                        rag_response = call_groq_api(rag_prompt)  # ✅ Groq API

                        st.write("**Context Used:**")
                        for i, doc in enumerate(retrieved_docs):
                            st.text_area(f"Context {i+1}", doc[:200] + "...", height=80, disabled=True, key=f"rag_context_{i}")

                        st.write("**Response:**")
                        st.write(rag_response)
                    else:
                        st.warning("No context found for RAG")

            with col2:
                st.subheader("🎯 Direct LLM Response")
                with st.spinner("Generating direct response..."):
                    direct_response = call_groq_api(comparison_question)  # ✅ Groq API
                    st.write("**Response:**")
                    st.write(direct_response)
        else:
            st.error("Please enter a question for comparison")

# ===============================
# Sidebar Info
# ===============================
total_docs = collection.count()
st.sidebar.markdown(f"**Knowledge Base:** {total_docs} documents")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Llama3-8b (Groq)")
st.sidebar.markdown("**API:** Groq (Free Tier)")