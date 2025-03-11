"""
Multi-Platform Translation Service
--------------------------------
This module provides a Streamlit-based web application that combines multiple
translation services (OpenAI, DeepL, and Google Translate) into a single interface.

Features:
- Translation caching
- Rate limiting
- Text chunking
- Asynchronous operations
- Error handling and logging
- Translation history
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass
import streamlit as st
from openai import OpenAI
import deepl
from googletrans import Translator
import asyncio
from concurrent.futures import ThreadPoolExecutor
import functools
import time
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TranslationConfig:
    """Configuration settings for translation services"""
    source_lang: str = "auto"
    cache_duration: int = 3600  # Cache duration in seconds
    rate_limit_delay: float = 0.5  # Delay between requests in seconds
    max_retries: int = 3  # Maximum number of retries for failed translations
    chunk_size: int = 1000  # Maximum characters per translation request

# Language code mappings for different translation services
LANGUAGE_CODES: Dict[str, Dict[str, str]] = {
    "Chinese (Simplified)": {"google": "zh-cn", "deepl": "ZH", "openai": "Chinese"},
    "English": {"google": "en", "deepl": "EN-US", "openai": "English"},
    "French": {"google": "fr", "deepl": "FR", "openai": "French"},
    "German": {"google": "de", "deepl": "DE", "openai": "German"},
    "Japanese": {"google": "ja", "deepl": "JA", "openai": "Japanese"},
    "Korean": {"google": "ko", "deepl": "KO", "openai": "Korean"},
    "Spanish": {"google": "es", "deepl": "ES", "openai": "Spanish"}
}

class TranslationCache:
    """
    Cache system for translation results to minimize API calls
    
    Attributes:
        cache: Dictionary storing translation results and timestamps
        cache_duration: How long to keep translations in cache (seconds)
    """
    
    def __init__(self, cache_duration: int = 3600):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_duration = cache_duration
    
    def get_cache_key(self, text: str, source_lang: str, target_lang: str, service: str) -> str:
        """Generate a unique cache key for a translation request"""
        return f"{hash(text)}:{source_lang}:{target_lang}:{service}"
    
    def get(self, text: str, source_lang: str, target_lang: str, service: str) -> Optional[str]:
        """Retrieve translation from cache if available and not expired"""
        key = self.get_cache_key(text, source_lang, target_lang, service)
        if key in self.cache:
            entry = self.cache[key]
            if datetime.now() - entry['timestamp'] < timedelta(seconds=self.cache_duration):
                logger.info(f"Cache hit for {service} translation")
                return entry['translation']
            else:
                del self.cache[key]
        return None
    
    def set(self, text: str, source_lang: str, target_lang: str, service: str, translation: str):
        """Store translation result in cache"""
        key = self.get_cache_key(text, source_lang, target_lang, service)
        self.cache[key] = {
            'translation': translation,
            'timestamp': datetime.now()
        }

def rate_limit(min_delay: float):
    """
    Decorator to implement rate limiting between API calls
    
    Args:
        min_delay: Minimum time (seconds) between calls
    """
    last_call = {}
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            key = func.__name__
            if key in last_call:
                elapsed = time.time() - last_call[key]
                if elapsed < min_delay:
                    time.sleep(min_delay - elapsed)
            result = func(*args, **kwargs)
            last_call[key] = time.time()
            return result
        return wrapper
    return decorator

class TranslatorBase:
    """
    Base class for translation services providing common functionality
    
    Attributes:
        config: Translation configuration settings
        cache: Translation cache instance
    """
    
    def __init__(self, config: TranslationConfig):
        self.config = config
        self.cache = TranslationCache(config.cache_duration)
    
    def split_text(self, text: str) -> list[str]:
        """
        Split text into smaller chunks for translation
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in text.split('. '):
            sentence = sentence.strip() + '. '
            sentence_length = len(sentence)
            
            if current_length + sentence_length > self.config.chunk_size:
                chunks.append(''.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks

class OpenAITranslator(TranslatorBase):
    """OpenAI GPT-based translation service implementation"""
    
    @rate_limit(min_delay=0.5)
    def translate(self, text: str, target_lang: str, api_key: str) -> str:
        """
        Translate text using OpenAI's API
        
        Args:
            text: Text to translate
            target_lang: Target language code
            api_key: OpenAI API key
            
        Returns:
            Translated text or error message
        """
        try:
            # Check cache first
            cached = self.cache.get(text, self.config.source_lang, target_lang, 'openai')
            if cached:
                return cached

            if not api_key or api_key.strip() == "":
                return "Please enter your OpenAI API key"
            
            client = OpenAI(api_key=api_key)
            chunks = self.split_text(text)
            translations = []
            
            for chunk in chunks:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": f"Translate the following text into {target_lang}."},
                        {"role": "user", "content": chunk}
                    ]
                )
                translations.append(response.choices[0].message.content)
            
            result = ' '.join(translations)
            self.cache.set(text, self.config.source_lang, target_lang, 'openai', result)
            return result
            
        except Exception as e:
            logger.error(f"OpenAI Translation Error: {str(e)}")
            if "insufficient_quota" in str(e):
                return "OpenAI API quota exceeded. Please check your billing details."
            elif "invalid_api_key" in str(e):
                return "Invalid OpenAI API key. Please check your API key."
            else:
                return f"OpenAI Translation Error: {str(e)}"

class DeepLTranslator(TranslatorBase):
    """DeepL translation service implementation"""
    
    @rate_limit(min_delay=0.3)
    def translate(self, text: str, target_lang: str, api_key: str) -> str:
        """
        Translate text using DeepL's API
        
        Args:
            text: Text to translate
            target_lang: Target language code
            api_key: DeepL API key
            
        Returns:
            Translated text or error message
        """
        try:
            # Check cache first
            cached = self.cache.get(text, self.config.source_lang, target_lang, 'deepl')
            if cached:
                return cached

            if not api_key or api_key.strip() == "":
                return "Please enter your DeepL API key"
            
            translator = deepl.Translator(api_key)
            chunks = self.split_text(text)
            translations = []
            
            for chunk in chunks:
                result = translator.translate_text(chunk, target_lang=target_lang)
                translations.append(result.text)
            
            result = ' '.join(translations)
            self.cache.set(text, self.config.source_lang, target_lang, 'deepl', result)
            return result
            
        except Exception as e:
            logger.error(f"DeepL Translation Error: {str(e)}")
            if "Authentication failed" in str(e):
                return "Invalid DeepL API key. Please check your API key."
            elif "quota exceeded" in str(e):
                return "DeepL API quota exceeded. Please check your usage limits."
            else:
                return f"DeepL Translation Error: {str(e)}"

class GoogleTranslator(TranslatorBase):
    """Google translate service implementation with async support"""
    
    async def translate_chunk(self, chunk: str, target_lang: str) -> str:
        """
        Translate a single chunk of text
        
        Args:
            chunk: Text chunk to translate
            target_lang: Target language code
            
        Returns:
            Translated chunk or error message
        """
        try:
            translator = Translator()
            result = await translator.translate(chunk, dest=target_lang)
            return result.text if result and hasattr(result, 'text') else "Translation failed"
        except Exception as e:
            logger.error(f"Google Translation Chunk Error: {str(e)}")
            return f"Translation error: {str(e)}"
    
    @rate_limit(min_delay=0.2)
    async def translate_async(self, text: str, target_lang: str) -> str:
        """
        Asynchronous Google translation implementation
        
        Args:
            text: Text to translate
            target_lang: Target language code
            
        Returns:
            Translated text or error message
        """
        try:
            # Check cache first
            cached = self.cache.get(text, self.config.source_lang, target_lang, 'google')
            if cached:
                return cached

            chunks = self.split_text(text)
            tasks = [self.translate_chunk(chunk, target_lang) for chunk in chunks]
            translations = await asyncio.gather(*tasks)
            
            result = ' '.join(translations)
            self.cache.set(text, self.config.source_lang, target_lang, 'google', result)
            return result
            
        except Exception as e:
            logger.error(f"Google Translation Error: {str(e)}")
            return f"Google Translation Error: {str(e)}"

def run_async_translation(text: str, target_lang: str, config: TranslationConfig) -> str:
    """Helper function to run async translation in a sync context"""
    translator = GoogleTranslator(config)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        result = loop.run_until_complete(translator.translate_async(text, target_lang))
        return result
    finally:
        loop.close()

def save_translation_history(text: str, translations: Dict[str, str]) -> None:
    """
    Save translation history to a JSON file
    
    Args:
        text: Original text
        translations: Dictionary of translations from different services
    """
    history_file = Path("translation_history.json")
    
    try:
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append({
            'timestamp': datetime.now().isoformat(),
            'original_text': text,
            'translations': translations
        })
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        logger.error(f"Error saving translation history: {str(e)}")

def main():
    """Main application function implementing the Streamlit interface"""
    st.set_page_config(
        page_title="Multi-Language Translation Platform",
        layout="wide"
    )

    # Initialize configuration and translation services
    config = TranslationConfig()
    openai_translator = OpenAITranslator(config)
    deepl_translator = DeepLTranslator(config)

    # Initialize session state for API keys
    for key in ["OPENAI_API", "DeeplAPI"]:
        if key not in st.session_state:
            st.session_state[key] = ""

    # Sidebar for settings
    with st.sidebar:
        st.session_state["OPENAI_API"] = st.text_input(
            label='OpenAI API Key',
            placeholder='Enter Your OpenAI API Key',
            value='',
            type='password'
        )
        st.markdown('---')
        st.session_state["DeeplAPI"] = st.text_input(
            label='DeepL API Key',
            placeholder='Enter Your DeepL API Key',
            value='',
            type='password'
        )
        st.markdown('---')
        target_language = st.selectbox(
            "Select Target Language",
            options=list(LANGUAGE_CODES.keys()),
            index=0
        )

    # Main interface
    st.header('Compare 3 different types of Translators')
    st.info('Note: Only OpenAI and DeepL translations require API keys. Google Translate is free!')
    st.markdown('---')
    
    st.subheader("Please input text here for the translation")
    txt = st.text_area(label="", placeholder="Input text to translate...", height=200)
    
    if txt and txt.strip():
        st.markdown('---')
        lang_codes = LANGUAGE_CODES[target_language]
        
        # Store translations for history
        translations = {}

        # Create three columns for translation results
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("ChatGPT Translation")
            if st.session_state["OPENAI_API"]:
                with st.spinner("Translating..."):
                    result = openai_translator.translate(
                        txt, 
                        lang_codes["openai"], 
                        st.session_state["OPENAI_API"]
                    )
                    translations['openai'] = result
                    st.info(result)
            else:
                st.error("OpenAI API key required")

        with col2:
            st.subheader("DeepL Translation")
            if st.session_state["DeeplAPI"]:
                with st.spinner("Translating..."):
                    result = deepl_translator.translate(
                        txt, 
                        lang_codes["deepl"],
                        st.session_state["DeeplAPI"]
                    )
                    translations['deepl'] = result
                    st.info(result)
            else:
                st.error("DeepL API key required")

        with col3:
            st.subheader("Google Translation")
            with st.spinner("Translating..."):
                with ThreadPoolExecutor() as executor:
                    result = executor.submit(
                        run_async_translation, 
                        txt, 
                        lang_codes["google"],
                        config
                    ).result()
                    translations['google'] = result
                    st.info(result)

        # Save translation history
        if translations:
            save_translation_history(txt, translations)

    else:
        st.info("Please enter some text to translate")

if __name__ == "__main__":
    main()