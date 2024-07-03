# with Chromadb and FastEmbedEmbeddings(slow response due to mistral model..!)
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="mistral")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
             maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.0,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None





# with faiss and HuggingFaceEmbeddings (fast response..!)
#import os
#import faiss
#from dotenv import load_dotenv
#from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.schema.output_parser import StrOutputParser
#from langchain.llms import HuggingFaceHub
#from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain
#
#
#load_dotenv()
#
#class ChatPDF:
#    def __init__(self):
#        self.llm = HuggingFaceHub(
#            repo_id="google/flan-t5-large",
#            model_kwargs={"max_length": 1024}
#        )
#        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
#        self.prompt = PromptTemplate.from_template(
#            """
#            <s> [INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context 
#            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
#            maximum and keep the answer concise. [/INST] </s> 
#            [INST] Question: {question} 
#            Context: {context} 
#            Answer: [/INST]
#            """
#        )
#        self.vector_store = None
#        self.retriever = None
#        self.chain = None
#
#    def ingest(self, pdf_file_path: str):
#        docs = PyPDFLoader(file_path=pdf_file_path).load()
#        chunks = self.text_splitter.split_documents(docs)
#        
#        embeddings = HuggingFaceEmbeddings()
#        vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)
#        self.vector_store = vector_store
#        self.retriever = self.vector_store.as_retriever(
#            search_type="similarity_score_threshold",
#            search_kwargs={
#                "k": 3,
#                "score_threshold": 0.0,
#            },
#        )
#
#        self.chain = LLMChain(
#            llm=self.llm,
#            prompt=self.prompt,
#            output_parser=StrOutputParser()
#        )
#
#    def ask(self, query: str):
#        if not self.chain or not self.retriever:
#            return "Please, add a PDF document first."
#        
#        search_results = self.retriever.get_relevant_documents(query)
#        context = "\n\n".join([doc.page_content for doc in search_results])
#        
#        return self.chain.run({"context": context, "question": query})
#
#    def clear(self):
#        self.vector_store = None
#        self.retriever = None
#        self.chain = None
#
