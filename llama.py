import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq  # Import Groq API integration

# Load API key from environment variables
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please provide responses to the user's queries."),
        ("user", "Question: {question}"),
    ]
)

def generate_response(question, api_key, engine, temperature, max_tokens):
    try:
        llm = ChatGroq(model=engine, api_key=api_key, temperature=temperature, max_tokens=max_tokens)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        answer = chain.invoke({'question': question})
        return answer
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("Enhanced Q&A Chatbot with Groq's Models")

# Sidebar Settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter Your Groq API Key:", type="password", value=API_KEY)

# Model selection (Groq supports Llama models)
# Model selection (Only use Groq-supported models)
engine = st.sidebar.selectbox("Select Model", [
    "llama-3.3-70b-versatile",  # Llama 3.3 70B Versatile Model
    "llama-3.1-8b-instant",     # Llama 3.1 8B Instant Model
    "mixtral-8x7b-32768",       # Mixtral 8x7B with 32K context window
    "gemma2-9b-it",             # Gemma 2 9B Italian Model
    "qwen-qwq-32b",             # Qwen QWQ 32B Model
    "qwen-2.5-coder-32b",       # Qwen 2.5 Coder 32B Model
    "qwen-2.5-32b",             # Qwen 2.5 32B Model
    "deepseek-r1-distill-qwen-32b",  # DeepSeek Distilled Qwen 32B Model
    "deepseek-r1-distill-llama-70b"  # DeepSeek Distilled Llama 70B Model
])



# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main chat interface
st.write("Go ahead and ask any question:")
user_input = st.text_input("You: ")

if user_input and api_key:
    response = generate_response(user_input, api_key, engine, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.warning("Please enter the Groq API key in the sidebar.")
else:
    st.write("Please provide the user input")
