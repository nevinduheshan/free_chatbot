import streamlit as st
import os
import chromadb  # 🔌 Pinecone වෙනුවට ChromaDB
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 🛠️ 1. අපේම Local ChromaDB එකට කනෙක්ට් වීම
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="financial_docs")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Local Financial AI Assistant", page_icon="📊", layout="centered")

st.markdown("""
    <style>
    .main-title { font-size: 40px; font-weight: 700; color: #1E3A8A; text-align: center; margin-bottom: 10px; }
    .sub-title { font-size: 18px; color: #4B5563; text-align: center; margin-bottom: 30px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Local Financial AI Analyst</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">100% Free Local Vector DB with Predictive Analysis</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("System Status")
    st.success("⚡ Local ChromaDB: Connected")
    st.success("⚡ Gemini: Connected")
    st.caption("Powered by SmartyAI Local Engine")

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your Local AI Financial Analyst. Ask me anything about the ingested PDFs!"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Core RAG Function using Local ChromaDB ---
def process_ai_response(chat_history):
    user_query = chat_history[-1]["content"]
    
    # 🛠️ 2. Pinecone වෙනුවට Local ChromaDB එකෙන් ප්‍රශ්නයට අදාළ PDF කෑලි සර්ච් කිරීම
    results = collection.query(
        query_texts=[user_query],
        n_results=5  # ප්‍රශ්නයට ගැලපෙනම හොඳම පේළි/කෑලි 5ක් සොයන්න
    )
    
    # ලැබුණු දත්ත එකතු කර text එකක් සෑදීම
    retrieved_chunks = results['documents'][0]
    raw_pdf_data = "\n\n".join(retrieved_chunks)
    
    conversation_context = ""
    for msg in chat_history[:-1]:
        author = "User" if msg["role"] == "user" else "Analyst"
        conversation_context += f"{author}: {msg['content']}\n"
        
    # 3. Advanced Predictive Prompt
    master_prompt = f"""
    You are an Elite Financial Analyst and Predictive Forecasting Expert. 
    Address the user's latest question using the conversation history and the locally retrieved PDF context.

    CRITICAL CAPABILITY (FINANCIAL FORECASTING):
    If the user asks for predictions, forecasts, or future years:
    1. Analyze the historical trends available in the [Retrieved Context from PDF].
    2. Project reasonable future metrics and explain your calculation logic.
    3. Include a "Key Assumptions & Risk Factors" section.

    [Conversation History]
    {conversation_context}
    
    [Retrieved Context from PDF]
    {raw_pdf_data}
    
    [Latest User Question]
    "{user_query}"
    
    Format professionally using clear markdown headings, bold text, and bullet points.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    analysis_response = model.generate_content(master_prompt)
    
    return analysis_response.text

# --- Handle User Input ---
if user_input := st.chat_input("Type your question here..."):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n[INFO] {current_time} - 📥 USER QUESTION: {user_input}")
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Searching local DB & generating response..."):
            try:
                ai_response = process_ai_response(st.session_state.messages)
                st.markdown(ai_response)
                
                print(f"[INFO] {datetime.now().strftime('%H:%M:%S')} - 📤 AI RESPONSE: Generated Successfully.")
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")