import re
import os
from typing import Optional, Any

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def extract_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from standard and short YouTube URLs."""
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        return video_id_match.group(1)
    return None

def fetch_transcript(video_id: str) -> Optional[str]:
    """Fetches the english transcript from a Youtube video id."""
    try:
        transcript_list = YouTubeTranscriptApi().list(video_id)
        # 1. Try manual or generated english
        try:
            fetched = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
        except Exception:
            try:
                fetched = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
            except Exception:
                # 2. Fall back to any language, translate to English
                base_transcript = transcript_list.find_transcript(transcript_list._find_manually_created_language() or transcript_list._find_generated_language())
                fetched = base_transcript.translate('en')

        fetched_transcript = fetched.fetch()
        transcript = " ".join(chunk.text for chunk in fetched_transcript)
        return transcript, None
    except TranscriptsDisabled:
        return None, "Transcripts Disabled"
    except Exception as e:
        return None, str(e)


def build_vector_store(transcript: str) -> FAISS:
    """Splits transcript and builds an in-memory FAISS vector store."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    # HuggingFace Embeddings (all-MiniLM-L6-v2) for local fast embedding
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def format_doc(retriever_docs):
    """Formats retrieved document chunks."""
    context_text = "\n\n".join(doc.page_content for doc in retriever_docs)
    return context_text

def get_rag_chain(vector_store: FAISS, gemini_api_key: str):
    """Returns the conversational Langchain runnable."""
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 4})
    
    llm = ChatGoogleGenerativeAI(
        model='gemini-3-flash-preview', 
        api_key=gemini_api_key, 
        temperature=0.2
    )

    prompt = PromptTemplate(template="""
You are an expert AI assistant designed to help users interact with a YouTube video based on its transcript.
Answer the user's question explicitly from the provided transcript context. 
If the context is insufficient, simply state that the video doesn't provide enough information to answer. 
Write engagingly, cleanly formatting your response with markdown.

Context:
{context}

Question: {question}
    """, input_variables=['context', 'question'])

    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_doc),
        'question': RunnablePassthrough()
    })
    
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    return main_chain
