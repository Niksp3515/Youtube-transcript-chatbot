import streamlit as st
import time
from chatbot import extract_video_id, fetch_transcript, build_vector_store, get_rag_chain

st.set_page_config(page_title="TubeChat AI", page_icon="🎬", layout="wide")


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

with st.sidebar:
    st.markdown("<h2 class='gradient-text'>⚙️ Configure Bot</h2>", unsafe_allow_html=True)
    st.write("Connect the model with your data.")
    
    gemini_api_key = st.text_input("Gemini API Key", type="password", placeholder="AIzaSy...")
    st.caption("Needed to initialize Google GenAI")
    
    st.markdown("---")
    
    youtube_url = st.text_input("YouTube Video URL", placeholder="https://youtu.be/...")
    
    if st.button("Process Video"):
        if not gemini_api_key:
            st.error("Please provide your Gemini API key first.")
        elif not youtube_url:
            st.error("Please enter a valid YouTube video URL.")
        else:
            with st.spinner("Extracting Video ID..."):
                time.sleep(0.5) 
                video_id = extract_video_id(youtube_url)
            
            if not video_id:
                st.error("Invalid YouTube URL! Please ensure it's a valid link.")
            else:
                st.session_state.current_video = video_id
                
                with st.spinner("Fetching English transcript..."):
                    transcript = fetch_transcript(video_id)
                
                if not transcript:
                    st.error("Could not fetch transcript. It might be disabled for this video, or no English captions exist.")
                else:
                    with st.spinner("Embedding and Building Knowledge Base..."):
                        vector_store = build_vector_store(transcript)
                        st.session_state.rag_chain = get_rag_chain(vector_store, gemini_api_key)
                        
                    st.success("✅ Video Processed Successfully! You can now ask questions about the video.")
                    st.session_state.messages = [{"role": "assistant", "content": "Hello! I've studied the video. What would you like to know?"}]



st.markdown("<h1 class='gradient-text'>🎬 TubeChat AI</h1>", unsafe_allow_html=True)
st.write("Your Intelligent YouTube Assistant. Interact seamlessly with video content using RAG.")

if not st.session_state.rag_chain:
    
    st.info("👈 Please enter your Gemini API Key and a YouTube URL in the sidebar to get started.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the video..."):
    if not st.session_state.rag_chain:
        st.warning("Please process a video in the sidebar first!")
    else:
        
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
