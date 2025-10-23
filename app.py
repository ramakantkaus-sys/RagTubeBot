import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="YouTube Transcript Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin-bottom: 1rem;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "transcript_loaded" not in st.session_state:
    st.session_state.transcript_loaded = False

def extract_video_id(url_or_id):
    """Extract video ID from YouTube URL or return the ID if already provided."""
    # Pattern for various YouTube URL formats
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'^([a-zA-Z0-9_-]{11})$'  # Direct video ID
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url_or_id)
        if match:
            return match.group(1)
    return None

def fetch_transcript(video_id):
    """Fetch transcript from YouTube video."""
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
        transcript = " ".join(snippet.text for snippet in fetched_transcript)
        return transcript, None
    except Exception as e:
        return None, str(e)

def create_rag_chain(transcript):
    """Create RAG chain from transcript."""
    try:
        # Split transcript into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Set up LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Define prompt template
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
Answer ONLY from the provided YouTube transcript context below.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Create RAG chain
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain, None
    except Exception as e:
        return None, str(e)

# Header
st.markdown('<div class="main-header">🎥 YouTube Transcript Chatbot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask questions about any YouTube video using AI</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key status
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.success("✅ OpenAI API Key loaded")
    else:
        st.error("❌ OpenAI API Key not found in .env")
    
    st.divider()
    
    # Video input
    st.subheader("📹 Load Video")
    video_input = st.text_input(
        "Enter YouTube URL or Video ID",
        placeholder="https://youtube.com/watch?v=... or video_id",
        help="Paste a YouTube URL or just the video ID"
    )
    
    load_button = st.button("🔄 Load Transcript", type="primary", use_container_width=True)
    
    if load_button and video_input:
        video_id = extract_video_id(video_input)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL or Video ID")
        else:
            with st.spinner("Fetching transcript..."):
                transcript, error = fetch_transcript(video_id)
                
                if error:
                    st.error(f"❌ Error: {error}")
                    st.session_state.transcript_loaded = False
                else:
                    with st.spinner("Creating RAG chain..."):
                        rag_chain, error = create_rag_chain(transcript)
                        
                        if error:
                            st.error(f"❌ Error creating RAG chain: {error}")
                            st.session_state.transcript_loaded = False
                        else:
                            st.session_state.rag_chain = rag_chain
                            st.session_state.video_id = video_id
                            st.session_state.transcript_loaded = True
                            st.session_state.messages = []  # Clear previous chat
                            st.success(f"✅ Loaded video: {video_id}")
                            st.rerun()
    
    st.divider()
    
    # Current video info
    if st.session_state.transcript_loaded:
        st.subheader("📊 Current Video")
        st.info(f"**Video ID:** {st.session_state.video_id}")
        st.markdown(f"[🔗 Watch on YouTube](https://youtube.com/watch?v={st.session_state.video_id})")
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    st.divider()
    
    # Sample questions
    st.subheader("💡 Sample Questions")
    sample_questions = [
        "Can you summarize the video?",
        "What are the main topics discussed?",
        "Who are the people mentioned?",
        "What are the key takeaways?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question}", use_container_width=True):
            if st.session_state.transcript_loaded:
                st.session_state.messages.append({"role": "user", "content": question})
                st.rerun()
    
    st.divider()
    
    # About
    with st.expander("ℹ️ About"):
        st.markdown("""
        **YouTube Transcript Chatbot**
        
        This app uses:
        - 🎥 YouTube Transcript API
        - 🤖 OpenAI GPT-4o-mini
        - 🔍 FAISS vector search
        - 🦜 LangChain RAG
        
        Built with Streamlit
        """)

# Main chat interface
if not st.session_state.transcript_loaded:
    st.info("👈 **Get started:** Enter a YouTube URL in the sidebar and click 'Load Transcript'")
    
    # Example section
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 1️⃣ Load Video")
        st.write("Enter any YouTube URL or video ID in the sidebar")
    
    with col2:
        st.markdown("### 2️⃣ Wait for Processing")
        st.write("The app will fetch and analyze the transcript")
    
    with col3:
        st.markdown("### 3️⃣ Ask Questions")
        st.write("Chat with the AI about the video content")
    
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Made with ❤️ using Streamlit | "
    "Powered by OpenAI & LangChain</div>",
    unsafe_allow_html=True
)
