"""
YouTube Video Summarizer
A Streamlit application that generates summaries of YouTube video transcripts using OpenAI's GPT-3.5.
"""

import streamlit as st
from typing import Optional, Dict, Any
import re
from dataclasses import dataclass
from functools import lru_cache
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI

# Constants
YOUTUBE_URL_PATTERN = r'^https:\/\/(www\.)?(youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]+)'
MAX_TOKENS = 3000
CHUNK_SIZE = 4000
MODEL_NAME = "gpt-3.5-turbo"
REQUEST_TIMEOUT = 120

@dataclass
class AppState:
    """Class to manage application state"""
    flag: bool = True
    openai_api: str = ""
    summary: str = ""

def init_session_state() -> None:
    """Initialize session state variables"""
    default_state = AppState()
    for key, value in default_state.__dict__.items():
        if key not in st.session_state:
            st.session_state[key] = value

@lru_cache(maxsize=100)
def validate_youtube_url(url: str) -> bool:
    """
    Validate YouTube URL format using regex pattern
    Args:
        url (str): YouTube URL to validate
    Returns:
        bool: True if valid, False otherwise
    """
    return bool(re.match(YOUTUBE_URL_PATTERN, url))

def create_llm(api_key: str) -> ChatOpenAI:
    """
    Create Language Model instance
    Args:
        api_key (str): OpenAI API key
    Returns:
        ChatOpenAI: Configured language model
    """
    return ChatOpenAI(
        temperature=0,
        openai_api_key=api_key,
        max_tokens=MAX_TOKENS,
        model_name=MODEL_NAME,
        request_timeout=REQUEST_TIMEOUT,
    )

def setup_prompts() -> tuple[PromptTemplate, PromptTemplate]:
    """
    Create prompt templates for summarization
    Returns:
        tuple: (map_prompt, combine_prompt)
    """
    map_prompt = PromptTemplate(
        template="Summarize the YouTube video transcript provided: ```{text}```",
        input_variables=["text"]
    )
    combine_prompt = PromptTemplate(
        template="Combine all provided transcripts into a concise summary: ```{text}```",
        input_variables=["text"]
    )
    return map_prompt, combine_prompt

def process_transcript(transcript: list, llm: ChatOpenAI) -> str:
    """
    Process transcript and generate summary
    Args:
        transcript (list): Video transcript
        llm (ChatOpenAI): Language model instance
    Returns:
        str: Generated summary
    """
    map_prompt, combine_prompt = setup_prompts()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    text = text_splitter.split_documents(transcript)
    
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        verbose=False,
        map_prompt=map_prompt,
        combine_prompt=combine_prompt
    )
    
    return chain.run(text)

def create_ui_layout() -> tuple[str, str]:
    """
    Create and setup UI layout
    Returns:
        tuple: (youtube_url, api_key)
    """
    st.set_page_config(page_title="YouTube Summarizer", layout="wide")
    st.header("📹 YouTube Scripts Summarizer")
    st.markdown('---')
    
    # Main content
    st.subheader("YouTube URL Input Here")
    youtube_url = st.text_input(" ", placeholder="https://www.youtube.com/watch?v=**********")
    
    # Sidebar
    with st.sidebar:
        api_key = st.text_input(
            label='OPENAI API Key',
            placeholder='Enter Your API Key',
            type='password'
        )
        st.markdown('---')
    
    return youtube_url, api_key

def handle_video_processing(youtube_url: str) -> Optional[Any]:
    """
    Handle video loading and display
    Args:
        youtube_url (str): YouTube video URL
    Returns:
        Optional[Any]: Video transcript or None if error
    """
    _, container, _ = st.columns([1, 6, 1])
    container.video(youtube_url)
    
    try:
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        return loader.load()
    except Exception as e:
        st.error(f"Error loading YouTube transcript: {e}")
        return None

def main() -> None:
    """Main application function"""
    init_session_state()
    youtube_url, api_key = create_ui_layout()
    
    if api_key:
        st.session_state.openai_api = api_key
    
    if len(youtube_url) > 2:
        if not validate_youtube_url(youtube_url):
            st.error("Invalid YouTube URL. Please check the format.")
            return
            
        transcript = handle_video_processing(youtube_url)
        if not transcript:
            return
            
        st.subheader("Summary Result")
        
        if st.session_state.flag:
            llm = create_llm(st.session_state.openai_api)
            st.session_state.summary = process_transcript(transcript, llm)
            st.session_state.flag = False
            
        st.success(st.session_state.summary)

if __name__ == "__main__":
    main()