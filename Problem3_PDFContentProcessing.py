"""
PDF Question Answering Application
This module provides functionality to load PDF documents and answer questions based on their content
using LangChain and OpenAI.

Features:
- PDF text extraction
- Text chunking and embedding
- Question answering using OpenAI
- Interactive Streamlit interface
"""
import streamlit as st
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from functools import lru_cache

# PDF processing
from PyPDF2 import PdfReader

# LangChain components
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import StreamlitCallbackHandler

@dataclass
class Config:
    """Application configuration settings"""
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MODEL_NAME: str = 'gpt-3.5-turbo'
    TEMPERATURE: float = 0
    MAX_TOKENS: int = 2000
    REQUEST_TIMEOUT: int = 120

class PDFProcessor:
    """Handles PDF file processing and text extraction"""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len
        )

    @staticmethod
    def read_pdf(pdf_file) -> str:
        """Extract text from PDF file"""
        pdf_reader = PdfReader(pdf_file)
        return "\n".join(
            page.extract_text() 
            for page in pdf_reader.pages
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks for processing"""
        return self.text_splitter.split_text(text)

class QASystem:
    """Handles question answering functionality"""
    
    def __init__(self, config: Config):
        self.config = config
        self.knowledge_base = None
        self.chain = None
        
    @lru_cache(maxsize=1)
    def setup_embeddings(self, api_key: str) -> OpenAIEmbeddings:
        """Initialize and cache embeddings"""
        return OpenAIEmbeddings(openai_api_key=api_key)
    
    def create_knowledge_base(self, chunks: List[str], api_key: str) -> None:
        """Create vector store knowledge base"""
        embeddings = self.setup_embeddings(api_key)
        self.knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    def setup_qa_chain(self, api_key: str) -> None:
        """Initialize the QA chain with conversation memory"""
        llm = ChatOpenAI(
            temperature=self.config.TEMPERATURE,
            openai_api_key=api_key,
            max_tokens=self.config.MAX_TOKENS,
            model_name=self.config.MODEL_NAME,
            request_timeout=self.config.REQUEST_TIMEOUT,
            streaming=True
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.knowledge_base.as_retriever(),
            memory=memory,
            callbacks=[StreamlitCallbackHandler(st.container())]
        )

    def get_answer(self, question: str) -> Optional[str]:
        """Get answer for user question"""
        if not self.chain:
            return None
        return self.chain({"question": question})["answer"]

class PDFAnalyzerApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = Config()
        self.pdf_processor = PDFProcessor(self.config)
        self.qa_system = QASystem(self.config)
        
    def setup_page(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="PDF Analyzer",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def setup_sidebar(self) -> Optional[str]:
        """Setup sidebar with API key input"""
        with st.sidebar:
            api_key = st.text_input(
                label='OpenAI API Key',
                placeholder='Enter Your API Key',
                type='password'
            )
            if api_key:
                st.session_state["OPENAI_API"] = api_key
            st.markdown('---')
            return api_key
            
    def main(self):
        """Main application flow"""
        self.setup_page()
        api_key = self.setup_sidebar()
        
        st.header("PDF File Questions and Answers Program 📜")
        st.markdown('---')
        st.subheader("Drag and Drop your PDF File Here")
        
        pdf_file = st.file_uploader("", type="pdf")
        
        if pdf_file and api_key:
            # Process PDF
            text = self.pdf_processor.read_pdf(pdf_file)
            chunks = self.pdf_processor.split_text(text)
            
            # Initialize QA system
            self.qa_system.create_knowledge_base(chunks, api_key)
            self.qa_system.setup_qa_chain(api_key)
            
            st.markdown('---')
            st.subheader("Ask your question about the PDF content")
            
            # Question input and answering
            user_question = st.text_input(
                "Enter your question:",
                placeholder="What would you like to know about the document?"
            )
            
            if user_question:
                with st.spinner("Analyzing..."):
                    response = self.qa_system.get_answer(user_question)
                    if response:
                        st.info(response)

if __name__ == '__main__':
    app = PDFAnalyzerApp()
    app.main()