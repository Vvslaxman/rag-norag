# with rag and using FastEmbedEmbeddings for embeddings, chromadb for storing and chatOllama for model selection (Mistral)
import os
import tempfile
import streamlit as st
from streamlit_chat import message
from streamlit_extras.add_vertical_space import add_vertical_space
from rag import ChatPDF

st.set_page_config(page_title="ChatPDF")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 LLM Chat App - RAG')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [Mistral](https://ollama.com/library/mistral) LLM model
    - [Chromadb](https://www.trychroma.com/)  
    - [FastEmbedEmbeddings](https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.fastembed.FastEmbedEmbeddings.html)                    
    ''')
    add_vertical_space(5)
    st.write('Made with ❤️ by [Lucky Empire](https://github.com/Vvslaxman)')

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
            agent_text = agent_text if agent_text else "I couldn't generate a response."

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def page():
    print("Starting Streamlit app...")
    import sys
    print("sys.path:", sys.path)

    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()

    st.header("ChatPDF")

    st.subheader("Upload a document")
    st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if "ingestion_spinner" not in st.session_state:
        st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()



# with rag using Hugging face for embeddings, faiss-cpu for storing and HugginFaceHub LLM
#import os
#import tempfile
#import streamlit as st
#from streamlit_chat import message
#from rag import ChatPDF
#from dotenv import load_dotenv
#import pickle
#from PyPDF2 import PdfReader
#from streamlit_extras.add_vertical_space import add_vertical_space
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.llms import HuggingFaceHub
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
#from sklearn.feature_extraction.text import TfidfVectorizer
#import faiss
#
#st.set_page_config(page_title="ChatPDF")
#
## Sidebar contents
#with st.sidebar:
#    st.title('🤗💬 LLM Chat App - RAG')
#    st.markdown('''
#    ## About
#    This app is an LLM-powered chatbot built using:
#    - [Streamlit](https://streamlit.io/)
#    - [LangChain](https://python.langchain.com/)
#    - [RAG](https://python.langchain.com/v0.1/docs/use_cases/question_answering/)          
#    - [Hugging Face](https://huggingface.co/)LLM model
#    - [Faiss-cpu](https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/)
#    - [HuggingFaceEmbeddings](https://python.langchain.com/v0.2/docs/integrations/platforms/huggingface/)
#    ''')
#    add_vertical_space(5)
#    st.write('Made with ❤️ by [Lucky Empire](https://github.com/Vvslaxman)')
#
#load_dotenv()
#
#def embed_texts(texts):
#    vectorizer = TfidfVectorizer()
#    embeddings = vectorizer.fit_transform(texts).toarray()
#    return embeddings, vectorizer
#
#def display_messages():
#    st.subheader("Chat")
#    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
#        message(msg, is_user=is_user, key=str(i))
#    st.session_state["thinking_spinner"] = st.empty()
#
#def process_input():
#    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
#        user_text = st.session_state["user_input"].strip()
#        with st.session_state["thinking_spinner"], st.spinner("Thinking"):
#            query_embedding = st.session_state["vectorizer"].transform([user_text]).toarray()
#            distances, indices = st.session_state["index"].search(query_embedding, k=3)
#            docs = [st.session_state["chunks"][idx] for idx in indices[0]]
#            context = "\n\n".join(docs)
#           # huggingfacehub_api_token =st.secrets['HUGGINGFACEHUB_ACCESS_TOKEN']
#            huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#
#            if not huggingfacehub_api_token:
#                agent_text = "Hugging Face API token not found. Please set it in the .env file."
#            else:
#                llm = HuggingFaceHub(
#                    repo_id="google/flan-t5-large",
#                    huggingfacehub_api_token=huggingfacehub_api_token,
#                    model_kwargs={"max_length": 1024}
#                )
#                prompt_template = PromptTemplate(
#                    input_variables=["context", "question"],
#                    template="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#                )
#                chain = LLMChain(llm=llm, prompt=prompt_template)
#
#                try:
#                    agent_text = chain.run({"context": context, "question": user_text})
#                except Exception as e:
#                    agent_text = f"An error occurred: {e}"
#
#        st.session_state["messages"].append((user_text, True))
#        st.session_state["messages"].append((agent_text, False))
#
#def read_and_save_file():
#    st.session_state["assistant"].clear()
#    st.session_state["messages"] = []
#    st.session_state["user_input"] = ""
#
#    for file in st.session_state["file_uploader"]:
#        with tempfile.NamedTemporaryFile(delete=False) as tf:
#            tf.write(file.getbuffer())
#            file_path = tf.name
#
#        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
#            pdf_reader = PdfReader(file_path)
#            text = ""
#            for page in pdf_reader.pages:
#                text += page.extract_text()
#            text_splitter = RecursiveCharacterTextSplitter(
#                chunk_size=1000,
#                chunk_overlap=200,
#                length_function=len
#            )
#            chunks = text_splitter.split_text(text=text)
#            embeddings, vectorizer = embed_texts(chunks)
#            index = faiss.IndexFlatL2(embeddings.shape[1])
#            index.add(embeddings)
#            st.session_state["chunks"] = chunks
#            st.session_state["vectorizer"] = vectorizer
#            st.session_state["index"] = index
#
#        os.remove(file_path)
#
#def page():
#    if "messages" not in st.session_state:
#        st.session_state["messages"] = []
#    
#    if "assistant" not in st.session_state:
#        st.session_state["assistant"] = ChatPDF()
#
#    st.header("ChatPDF")
#
#    st.subheader("Upload a document")
#    st.file_uploader(
#        "Upload document",
#        type=["pdf"],
#        key="file_uploader",
#        on_change=read_and_save_file,
#        label_visibility="collapsed",
#        accept_multiple_files=True,
#    )
#
#    if "ingestion_spinner" not in st.session_state:
#        st.session_state["ingestion_spinner"] = st.empty()
#
#    display_messages()
#    st.text_input("Message", key="user_input", on_change=process_input)
#
#if __name__ == "__main__":
#    page()
#
