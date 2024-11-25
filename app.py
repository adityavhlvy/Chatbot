import os
import streamlit as st
from transformers import pipeline
from langchain.chains import RetrievalQA
from embedding import get_embedding_function
from langchain_community.llms import HuggingFacePipeline
from langchain_chroma import Chroma
from database import add_to_chroma, split_documents, load_documents

# Set the page layout to wide
st.set_page_config(layout="wide")

# Define constants and paths
# checkpoint = "MBZUAI/LaMini-Flan-T5-783M"
checkpoint = r"C:LLM-Model/LaMini-Flan-T5-783M"
offload_dir = "C:/LLM-Offload"

# CHROMA_PATH = "chroma"
# DATA_PATH = "data"

CHROMA_PATH = "chroma-new"
DATA_PATH = "data-new"

# Ensure offload directory exists
os.makedirs(offload_dir, exist_ok=True)

# In-memory users data (for demonstration purposes)
users = {"user1": "password1", "user2": "password2"}


def login(username, password):
    return users.get(username) == password


def display_login():
    # Display the login form
    username = st.sidebar.text_input("Username", key="login_username")
    password = st.sidebar.text_input("Password", type="password", key="login_password")

    if st.sidebar.button("Login", key="login_button"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Login successful!")
        else:
            st.sidebar.error("Invalid username or password")


@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        "text2text-generation",
        model=checkpoint,
        tokenizer=checkpoint,
        max_length=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm


@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever(search_kwargs={"k": 5})
    )
    return qa, db


def process_answer(instruction, chat_history):
    try:
        qa, db = qa_llm()
        full_instruction = instruction
        generated_text = qa(full_instruction)
        answer = generated_text.get("result", "I don't know.")
        results = db.similarity_search_with_score(instruction, k=5)

        relevant_chunks = []
        for doc, _score in results:
            doc_name = doc.metadata.get("source", "Unknown Document")
            chunk_content = f"**Document: {doc_name}**\n{doc.page_content}"
            relevant_chunks.append(chunk_content)

        return answer, generated_text, relevant_chunks
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return "I don't know.", None, []


def display_chat_history_and_chunks(relevant_chunks):
    cols = st.columns([4, 5])

    with cols[0]:
        for i in range(0, len(st.session_state.chat_history), 2):
            user_msg = st.session_state.chat_history[i]
            st.markdown(
                f"<div style='text-align: left; background-color: #4A5045; padding: 10px; border-radius: 10px; margin-bottom: 5px; color: #FFFFFF;'>{user_msg}</div>",
                unsafe_allow_html=True,
            )
            if i + 1 < len(st.session_state.chat_history):
                bot_msg = st.session_state.chat_history[i + 1]
                st.markdown(
                    f"<div style='text-align: right; background-color: #3E5337; padding: 10px; border-radius: 10px; margin-bottom: 5px; color: #FFFFFF;'>{bot_msg}</div>",
                    unsafe_allow_html=True,
                )

    with cols[1]:
        st.subheader("Relevant Chunks:")
        for chunk in relevant_chunks:
            # Extract the document name from the chunk content
            doc_name_start = chunk.find("**Document:") + len("**Document:")
            doc_name_end = chunk.find("**", doc_name_start)
            doc_name = chunk[doc_name_start:doc_name_end].strip()

            st.markdown(
                f"""
                <div style='background-color: #F0F0F0; padding: 15px; border-radius: 10px; margin-bottom: 10px; border-left: 5px solid #4A90E2;'>
                    <h4 style='margin-top: 0; color: #4A90E2;'>{doc_name}</h4>
                    <p style='color: #333333;'>{chunk}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def upload_and_process_pdf():
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        with open(os.path.join(DATA_PATH, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")
        st.info("Processing the PDF file...")
        documents = load_documents()
        chunks = split_documents(documents)
        add_to_chroma(chunks)
        st.success("PDF processed and added to the vectorbase!")


def main():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    st.sidebar.title("Menu")
    if st.session_state.logged_in:
        st.sidebar.text(f"Logged in as {st.session_state.username}")
        if st.sidebar.button("Logout", key="logout_button"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.sidebar.success("Logged out successfully!")
    else:
        display_login()

    st.title("Enhanced PDF Chatbot with RAG")
    with st.expander("About the App"):
        st.markdown(
            """
            This app allows you to ask questions about your documents using a local LLM model and retrieval-based QA.
            """
        )

    if st.session_state.logged_in:
        upload_and_process_pdf()
    else:
        st.warning("Please log in to upload documents.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    question = st.text_area("Enter Your Question")
    if st.button("Get Answer"):
        answer, generated_text, relevant_chunks = process_answer(
            question, st.session_state.chat_history
        )
        st.session_state.chat_history.append(f"Q: {question}")
        st.session_state.chat_history.append(f"A: {answer}")
        display_chat_history_and_chunks(relevant_chunks)

    if st.button("Reset Chat History"):
        st.session_state.chat_history = []
        st.success("Chat history has been reset.")


if __name__ == "__main__":
    main()
