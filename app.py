import streamlit as st
import time
from chatbot import extract_video_id, fetch_transcript, build_vector_store, get_rag_chain

st.set_page_config(page_title="TubeChat AI", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")


try:
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass


if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "current_video" not in st.session_state:
    st.session_state.current_video = None


# --- Header ---
st.markdown("<h1 class='gradient-text' style='text-align: center;'>🎬 TubeChat AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.1rem;'>Your Intelligent YouTube Assistant. Interact seamlessly with video content using RAG.</p>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.05); margin-bottom: 2rem;'>", unsafe_allow_html=True)

def process_video(youtube_url, gemini_api_key):
    if not gemini_api_key:
        st.error("Please provide your Gemini API key first.")
        return
    if not youtube_url:
        st.error("Please enter a valid YouTube video URL.")
        return
        
    with st.spinner("Extracting Video ID..."):
        time.sleep(0.5) 
        video_id = extract_video_id(youtube_url)
    
    if not video_id:
        st.error("Invalid YouTube URL! Please ensure it's a valid link.")
        return
        
    st.session_state.current_video = video_id
    
    with st.spinner("Fetching English transcript..."):
        transcript, error_msg = fetch_transcript(video_id)
    
    if not transcript:
        st.error(f"Could not fetch transcript. Technical details: {error_msg}")
    else:
        with st.spinner("Embedding and Building Knowledge Base..."):
            vector_store = build_vector_store(transcript)
            st.session_state.rag_chain = get_rag_chain(vector_store, gemini_api_key)
            
        st.success("✅ Video Processed Successfully! You can now ask questions about the video.")
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I've studied the video. What would you like to know?"}]
        st.rerun()

# --- Main Logic ---
if not st.session_state.rag_chain:
    # Centered Setup Screen
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.container(border=True):
            st.markdown("<h3 class='gradient-text' style='text-align: center;'>⚙️ Getting Started</h3>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Please enter your API Key and a YouTube URL below to begin chatting.</p>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            gemini_key_input = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIzaSy...", key="setup_key")
            yt_url_input = st.text_input("🔗 YouTube Video URL", placeholder="https://youtu.be/...", key="setup_url")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Process Video", use_container_width=True):
                process_video(yt_url_input, gemini_key_input)

else:
    # Sidebar only for changing/updating to a new video later
    with st.sidebar:
        st.markdown("<h3 class='gradient-text'>⚙️ Setup Another Video</h3>", unsafe_allow_html=True)
        gemini_new = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...", key="sidebar_key")
        yt_new = st.text_input("YouTube Video URL", placeholder="https://youtu.be/...", key="sidebar_url")
        if st.button("Process New Video", use_container_width=True):
            process_video(yt_new, gemini_new)

    # Chat Interface
    st.info(f"Currently discussing video ID: `{st.session_state.current_video}`")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error generating response: {e}")
