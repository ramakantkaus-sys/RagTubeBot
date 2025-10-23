# Required imports
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from dotenv import load_dotenv
import os

# ----------------- Step 0: Load environment -----------------
load_dotenv()

# ----------------- Step 1: Fetch transcript -----------------
video_id = "Gfr50f6ZBvo"  # only the ID, not full URL

try:
    # Create API instance and fetch transcript
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id, languages=["en"])
    transcript = " ".join(snippet.text for snippet in fetched_transcript)
    print("Transcript fetched successfully!")
except (TranscriptsDisabled, Exception) as e:
    print(f"Could not fetch transcript: {e}")
    transcript = ""

# ----------------- Step 2: Split transcript into chunks -----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

# ----------------- Step 3: Create embeddings and vector store -----------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.getenv("OPENAI_API_KEY"))
vector_store = FAISS.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# ----------------- Step 4: Set up LLM -----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# ----------------- Step 5: Define prompt template -----------------
prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant.
Answer ONLY from the provided YouTube transcript context below.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}

Answer:"""
)

# ----------------- Step 6: Create RAG chain manually -----------------
def format_docs(docs):
    """Combine retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)

# Build the RAG chain using LCEL (LangChain Expression Language)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ----------------- Step 7: Ask questions -----------------
print("\n" + "="*60)
print("YouTube Transcript Chatbot Ready!")
print("="*60 + "\n")

question_1 = "Is the topic of nuclear fusion discussed in this video? If yes, what was discussed?"
print(f"Q1: {question_1}")
answer_1 = rag_chain.invoke(question_1)
print(f"A1: {answer_1}\n")

question_2 = "Who is Demis?"
print(f"Q2: {question_2}")
answer_2 = rag_chain.invoke(question_2)
print(f"A2: {answer_2}\n")

question_3 = "Can you summarize the video?"
print(f"Q3: {question_3}")
answer_3 = rag_chain.invoke(question_3)
print(f"A3: {answer_3}\n")

print("="*60)
print("Done! You can modify the questions in main.py or build a loop.")
print("="*60)
