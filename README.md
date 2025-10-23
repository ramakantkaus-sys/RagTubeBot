# YouTube Transcript Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about YouTube videos using their transcripts and OpenAI's GPT models.

## Features

- **Fetch YouTube transcripts** automatically using video ID or URL
- **Semantic search** with FAISS vector store and OpenAI embeddings
- **Context-aware answers** using GPT-4o-mini
- **Beautiful Streamlit web UI** with chat interface
- **Simple CLI interface** (`main.py`) for quick testing

## Setup

### 1. Install Dependencies

```powershell
& "C:\Users\Ramakant\Desktop\youtube chatbot\venv\python.exe" -m pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Important:** 
- No quotes around the key
- No spaces around `=`
- Add `.env` to `.gitignore` (already done)
- **Never commit your API key to version control**

### 3. Run the Chatbot

#### Option A: Streamlit Web UI (Recommended)

```powershell
& "C:\Users\Ramakant\Desktop\youtube chatbot\venv\python.exe" -m streamlit run app.py
```

This will open a beautiful web interface in your browser where you can:
- Enter any YouTube URL or video ID
- Chat with the AI about the video
- See chat history
- Use sample questions

#### Option B: CLI Script

```powershell
& "C:\Users\Ramakant\Desktop\youtube chatbot\venv\python.exe" main.py
```

This runs the simple command-line version with hardcoded questions.

## How It Works

1. **Fetch transcript** from YouTube using `youtube-transcript-api`
2. **Split into chunks** (1000 chars, 200 overlap) using `RecursiveCharacterTextSplitter`
3. **Generate embeddings** with OpenAI's `text-embedding-3-small`
4. **Store in FAISS** vector database for fast similarity search
5. **Answer questions** by retrieving relevant chunks and using GPT-4o-mini

## Customize

### Change Video

Edit `main.py` line 17:

```python
video_id = "YOUR_VIDEO_ID"  # from youtube.com/watch?v=YOUR_VIDEO_ID
```

### Add More Questions

Add to the bottom of `main.py`:

```python
question_4 = "Your question here?"
print(f"Q4: {question_4}")
answer_4 = rag_chain.invoke(question_4)
print(f"A4: {answer_4}\n")
```

### Build Interactive Loop

Replace Step 7 in `main.py` with:

```python
print("\nYouTube Transcript Chatbot (type 'quit' to exit)\n")
while True:
    question = input("Ask a question: ")
    if question.lower() in ['quit', 'exit', 'q']:
        break
    answer = rag_chain.invoke(question)
    print(f"Answer: {answer}\n")
```

## Tech Stack

- **YouTube Transcript API** - Fetch captions
- **LangChain** (modular packages) - RAG framework
- **OpenAI** - Embeddings (text-embedding-3-small) + LLM (gpt-4o-mini)
- **FAISS** - Vector similarity search
- **Python 3.10** - Runtime

## Troubleshooting

### ModuleNotFoundError

Make sure you installed dependencies in the correct conda environment:

```powershell
& "C:\Users\Ramakant\Desktop\youtube chatbot\venv\python.exe" -m pip install -r requirements.txt
```

### No captions available

Some videos don't have transcripts. Try a different video ID.

### API Key Error

- Check `.env` file format (no quotes, no spaces)
- Verify key is valid in OpenAI dashboard
- Ensure `python-dotenv` is installed

## License

MIT
