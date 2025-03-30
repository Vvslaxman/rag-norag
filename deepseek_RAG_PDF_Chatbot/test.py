import os
import tempfile
import time
import streamlit as st
from langchain.chains import RetrievalQA, LLMChain
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.llms import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.schema.output_parser import StrOutputParser 
import faiss
import pyperclip  # to copy text to clipboard

# Initialize session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Deepseek R1-1.5B (RAG)"
if "prev_model" not in st.session_state:
    st.session_state.prev_model = "Deepseek R1-1.5B (RAG)"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "huggingface_chain" not in st.session_state:
    st.session_state.huggingface_chain = None

# Approach Details
approach_details = {
    "Deepseek R1-1.5B (RAG)": {
        "description": "*Retrieval-Augmented Generation (RAG)* with Deepseek R1-1.5B.",
        "tech_stack": "- *ChatOllama* for answering queries\n- *OllamaEmbeddings* for document embeddings\n- *Chroma Vector Store* for retrieval",
    },
    "HuggingFace Flan-T5 (Non-RAG)": {
        "description": "*Direct LLM Answering (Non-RAG)* with Flan-T5 large.",
        "tech_stack": "- *Hugging Face Flan-T5* for response generation\n- *FAISS Vector Store* for retrieval\n- *LLMChain* for query processing",
    },
}

# Sidebar for Model Selection & File Upload
with st.sidebar:
    st.markdown("## *Select Approach* 🎯")
    new_model = st.radio("Choose a model:", ["Deepseek R1-1.5B (RAG)", "HuggingFace Flan-T5 (Non-RAG)"])

    # Detect approach switch
    if new_model != st.session_state.selected_model:
        st.warning(f"⚠ You selected a different approach: *{st.session_state.selected_model} → {new_model}*")
        if st.button("✅ Confirm & Switch"):
            st.session_state.selected_model = new_model
            st.session_state.messages = []  # Clear past messages
            st.session_state.vector_store = None
            st.session_state.retriever = None
            st.session_state.qa_chain = None
            st.session_state.huggingface_chain = None
            st.rerun()  # ✅ Fixed: Correct method for reloading Streamlit

    # Show selected approach details
    st.markdown(f"### *🚀 {st.session_state.selected_model}*")
    st.info(approach_details[st.session_state.selected_model]["description"])
    st.markdown(f"#### *🛠 Tech Stack Used:*")
    st.markdown(approach_details[st.session_state.selected_model]["tech_stack"])

    # File Upload
    st.markdown("## *📄 Upload PDFs*")
    pdfs = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if st.button("📚 Create Knowledge Base"):
        if not pdfs:
            st.warning("⚠ Please upload PDF documents first!")
        else:
            with st.spinner("Processing documents... ⏳"):
                start_time = time.time()

                # *Step-by-Step Feedback for Knowledge Base Creation*
                # Step 1: Loading and parsing PDFs
                st.markdown("### Step 1: Loading and parsing PDFs 📄")
                pdf_paths = []
                with tempfile.TemporaryDirectory() as temp_dir:
                    for pdf in pdfs:
                        path = os.path.join(temp_dir, pdf.name)
                        with open(path, "wb") as f:
                            f.write(pdf.getbuffer())
                        pdf_paths.append(path)
                    
                    documents = []
                    for path in pdf_paths:
                        loader = PDFPlumberLoader(path)
                        documents.extend(loader.load())
                
                st.markdown("✅ PDFs loaded successfully!")

                # Step 2: Splitting Documents
                st.markdown("### Step 2: Splitting documents into chunks 🔄")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
                splits = text_splitter.split_documents(documents)
                st.markdown("✅ Documents split into chunks!")

                # Step 3: Create Embeddings and Vector Store
                if st.session_state.selected_model.startswith("Deepseek"):
                    st.markdown("### Step 3: Creating embeddings using Ollama 🧠")
                    embeddings = OllamaEmbeddings(model="nomic-embed-text")
                    vector_store = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
                    st.session_state.vector_store = vector_store
                    st.session_state.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
                    st.markdown("✅ Embeddings created using Ollama!")

                else:  # HuggingFace
                    st.markdown("### Step 3: Creating embeddings using HuggingFace 🧠")
                    pdf_text = ""
                    for pdf in pdfs:
                        pdf_reader = PdfReader(pdf)
                        pdf_text += "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
                    chunks = text_splitter.split_text(pdf_text)
                    embeddings = HuggingFaceEmbeddings()
                    faiss_index = FAISS.from_texts(chunks, embeddings)
                    st.session_state.retriever = faiss_index.as_retriever()
                    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    
                    # *Create a Prompt Template* for HuggingFace
                    prompt_template = PromptTemplate(
                        input_variables=["context", "question"],
                        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    )

                    # *Initialize HuggingFace Chain*
                    huggingface_llm = HuggingFaceHub(repo_id="google/flan-t5-large", huggingfacehub_api_token=huggingfacehub_api_token, model_kwargs={"temperature": 0.5, "max_length": 512})
                    huggingface_chain = LLMChain(llm=huggingface_llm, prompt=prompt_template, output_parser=StrOutputParser())
                    st.session_state.huggingface_chain = huggingface_chain
                    
                    st.markdown("✅ Embeddings created using HuggingFace!")

                elapsed_time = time.time() - start_time
                st.success(f"✅ Knowledge Base Created in {elapsed_time:.2f} seconds")

# Chat Interface
st.title("💬 ChatPDF.ai")
st.markdown(f"*Using {st.session_state.selected_model}*")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "<think>" in message["content"]:  # ✅ Highlight <think>...</think>
            formatted_content = message["content"].replace("<think>", "<span style='color: blue;'>🧠 ").replace("</think>", "</span>")
            st.markdown(formatted_content, unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("🔎 Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Fetching answer... ⏳"):
            start_time = time.time()
            if st.session_state.selected_model.startswith("Deepseek"):
                if st.session_state.vector_store:
                    llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.3)
                    qa_chain = RetrievalQA.from_chain_type(llm, retriever=st.session_state.retriever, chain_type="stuff")
                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]
                else:
                    answer = "❗ Please create a knowledge base first."

            else:  # HuggingFace (Non-RAG)
                if st.session_state.huggingface_chain:
                    search_results = st.session_state.retriever.get_relevant_documents(prompt)
                    context = "\n\n".join([doc.page_content for doc in search_results])
                    answer = st.session_state.huggingface_chain.run({"context": context, "question": prompt}) 
                else:
                    answer = "❗ Please create a knowledge base first."

            elapsed_time = time.time() - start_time
            st.markdown(f"⏳ *Time Taken:* {elapsed_time:.2f} seconds")
            
            # Display answer
            if "<think>" in answer:
                answer = answer.replace("<think>", "🧠 ").replace("</think>", "")  # ✅ Highlight <think>...<think/>
                formatted_answer = f"<span style='color: blue;'>{answer}</span>"
            else:
                formatted_answer = answer

            # Display the response with the copy to clipboard button
            st.markdown(formatted_answer, unsafe_allow_html=True)

            # Copy to clipboard button
            st.button("Copy Answer to Clipboard", on_click=lambda: pyperclip.copy(answer))

    st.session_state.messages.append({"role": "assistant", "content": answer})

