import os
import tempfile
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
from langchain.schema.output_parser import StrOutputParser
from PyPDF2 import PdfReader
import faiss

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Deepseek R1-1.5B"
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "huggingface_chain" not in st.session_state:
    st.session_state.huggingface_chain = None

# Sidebar for model selection and file upload
with st.sidebar:
    st.markdown("## Select Model")
    model_choice = st.radio("Choose a model:", ["Deepseek R1-1.5B", "HuggingFace Flan-T5"])
    st.session_state.selected_model = model_choice
    st.markdown("## Upload PDFs")
    pdfs = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if st.button("Create Knowledge Base"):
        if not pdfs:
            st.warning("Please upload PDF documents first!")
        else:
            with st.spinner("Processing documents..."):
                if st.session_state.selected_model == "Deepseek R1-1.5B":
                    # Deepseek processing
                    with tempfile.TemporaryDirectory() as temp_dir:
                        pdf_paths = [os.path.join(temp_dir, pdf.name) for pdf in pdfs]
                        for pdf, path in zip(pdfs, pdf_paths):
                            with open(path, "wb") as f:
                                f.write(pdf.getbuffer())
                        
                        documents = []
                        for path in pdf_paths:
                            loader = PDFPlumberLoader(path)
                            documents.extend(loader.load())
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
                        splits = text_splitter.split_documents(documents)
                        embeddings = OllamaEmbeddings(model="nomic-embed-text")
                        vector_store = Chroma.from_documents(splits, embeddings, persist_directory="./chroma_db")
                        st.session_state.vector_store = vector_store
                        st.session_state.retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3})
                        st.session_state.qa_chain = None
                    
                else:
                    # HuggingFace processing
                    pdf_text = ""
                    for pdf in pdfs:
                        pdf_reader = PdfReader(pdf)
                        pdf_text += "\n".join([page.extract_text() for page in pdf_reader.pages])
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
                    chunks = text_splitter.split_text(pdf_text)
                    embeddings = HuggingFaceEmbeddings()
                    faiss_index = FAISS.from_texts(chunks, embeddings)
                    st.session_state.retriever = faiss_index.as_retriever()
                    
                    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                    if not huggingfacehub_api_token:
                        st.error("Hugging Face API token not found. Please set it in the environment variables.")
                    else:
                        llm = HuggingFaceHub(
                            repo_id="google/flan-t5-large",
                            huggingfacehub_api_token=huggingfacehub_api_token,
                            model_kwargs={"max_length": 1024}
                        )
                        prompt_template = PromptTemplate(
                            input_variables=["context", "question"],
                            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                        )
                        st.session_state.huggingface_chain = LLMChain(llm=llm, prompt=prompt_template, output_parser=StrOutputParser())
    
# Chat interface
st.title("ChatPDF.ai")
st.markdown(f"Using {st.session_state.selected_model}")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Fetching answer..."):
            if st.session_state.selected_model == "Deepseek R1-1.5B":
                if st.session_state.vector_store:
                    llm = ChatOllama(model="deepseek-r1:1.5b", temperature=0.3)
                    qa_chain = RetrievalQA.from_chain_type(llm, retriever=st.session_state.retriever, chain_type="stuff")
                    response = qa_chain.invoke({"query": prompt})
                    answer = response["result"]
                else:
                    answer = "Please create a knowledge base first."
            else:
                if st.session_state.huggingface_chain:
                    search_results = st.session_state.retriever.get_relevant_documents(prompt)
                    context = "\n\n".join([doc.page_content for doc in search_results])
                    answer = st.session_state.huggingface_chain.run({"context": context, "question": prompt})
                else:
                    answer = "Please create a knowledge base first."
            
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})