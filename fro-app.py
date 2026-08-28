import streamlit as st
import chromadb
import ollama

from chromadb.config import Settings
from chromadb.utils import embedding_functions


# -----------------------------
# Configuration
# -----------------------------

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "hair_care_insights"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = "llama3.2:1b"


# -----------------------------
# Page setup
# -----------------------------

st.set_page_config(
    page_title="Fro 🌱",
    page_icon="🌿",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown(
    """
    <style>

    body {
        background-color: #F7F5F0;
    }

    .stChatMessage {
        padding: 12px;
        border-radius: 14px;
        margin-bottom: 10px;
    }

    textarea:focus,
    input:focus {
        border-color: #2F6B4F !important;
        box-shadow: 0 0 0 1px #2F6B4F !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Header
# -----------------------------

st.markdown("## 🌱 Fro")

st.markdown(
    """
    <p style='color:#2F6B4F; font-size:16px;'>
        Real hair talk. Clear, caring, and grounded in knowledge.
    </p>
    """,
    unsafe_allow_html=True
)


# -----------------------------
# Session state
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# Load Chroma
# -----------------------------

@st.cache_resource
def load_chroma():

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(
            anonymized_telemetry=False
        )
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )


collection = load_chroma()


# -----------------------------
# Model warm-up
# -----------------------------

@st.cache_resource
def warm_up_model():

    ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": "hi"
            }
        ]
    )


warm_up_model()


# -----------------------------
# Fro system prompt
# -----------------------------

FRO_SYSTEM_PROMPT = """
You are a friendly, grounded guide for people with type 3–4 textured hair.

Your role is to answer hair-related questions clearly and naturally,
like a knowledgeable friend. You are supportive without being emotional,
and helpful without sounding like a lesson or checklist.

Language and tone rules:
- Always use gender-neutral language.
- Do not refer to yourself or your role.
- No pet names or hype.
- Calm, clear, and practical.
- Keep responses concise, generally 5–7 sentences unless asked for more.
"""


# -----------------------------
# Retrieve context
# -----------------------------

def retrieve_context(
    user_question,
    n_results=2
):
    results = collection.query(
        query_texts=[user_question],
        n_results=n_results
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    return "\n\n".join(documents)


# -----------------------------
# Stream Fro's answer
# -----------------------------

def stream_fro_answer(user_question):

    context = retrieve_context(
        user_question
    )

    messages = [
        {
            "role": "system",
            "content": FRO_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""
Use the following retrieved knowledge as context
for answering the question.

Context:
{context}

Question:
{user_question}
"""
        }
    ]

    response_text = ""

    for chunk in ollama.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True
    ):

        if "message" in chunk:
            token = chunk["message"]["content"]
            response_text += token
            yield response_text


# -----------------------------
# Display chat history
# -----------------------------

for message in st.session_state.messages:

    avatar = (
        "images/fro-user.png"
        if message["role"] == "user"
        else "images/fro.png"
    )

    with st.chat_message(
        message["role"],
        avatar=avatar
    ):
        st.markdown(
            message["content"]
        )


# -----------------------------
# User input
# -----------------------------

user_input = st.chat_input(
    "Ask Fro about your hair..."
)


if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message(
        "user",
        avatar="images/fro-user.png"
    ):
        st.markdown(user_input)

    with st.chat_message(
        "assistant",
        avatar="images/fro.png"
    ):

        placeholder = st.empty()
        full_response = ""

        for partial in stream_fro_answer(
            user_input
        ):
            full_response = partial
            placeholder.markdown(
                full_response
            )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )
