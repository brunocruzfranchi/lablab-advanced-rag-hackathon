import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st
from chain.rag_graph import ask_question, make_rag_chain
from data_processing.preprocess import create_documents
from data_processing.vector_database import create_vector_db
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredHTMLLoader
from unstructured.cleaners.core import clean_extra_whitespace

load_dotenv()

TMP_DIR = Path(__file__).resolve().parent.joinpath("data", "tmp")

if not TMP_DIR.exists():
    TMP_DIR.mkdir(parents=True)

LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Advanced RAG Hackathon - Medify")
st.title("Advanced RAG Hackathon - Medify")


def load_documents(file_path):

    loader = UnstructuredHTMLLoader(
        file_path=file_path, post_processors=[clean_extra_whitespace]
    )

    documents = loader.load()

    return documents


def split_documents(docs):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=10,
        length_function=len,
        is_separator_regex=False,
    )

    contents = []
    metadatas = []

    for doc in docs:
        for chunk in doc:
            contents.append(chunk.page_content)
            metadatas.append(chunk.metadata)

    texts = text_splitter.create_documents(contents, metadatas)

    return texts


def input_fields():
    st.session_state.source_docs = st.file_uploader(
        label="Upload Documents", type="html", accept_multiple_files=False
    )


def process_documents():
    try:
        if st.session_state.source_docs is not None:
            with NamedTemporaryFile(
                delete=False, dir=TMP_DIR.as_posix(), suffix=".html"
            ) as tmp_file:
                try:
                    bytes_data = st.session_state.source_docs.getvalue()
                    tmp_file.write(bytes_data)
                    tmp_path = tmp_file.name
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            documents = load_documents(tmp_path)

            for _file in TMP_DIR.iterdir():
                temp_file = TMP_DIR.joinpath(_file)
                temp_file.unlink()

            texts = []

            if documents:
                for document in documents:
                    texts.append(create_documents(document.page_content))

            texts = split_documents(texts)

            st.session_state.vector_db = create_vector_db(
                texts, st.session_state.database
            )

    except Exception as e:
        st.error(f"An error occurred: {e}")


def main_streamlit():

    with st.sidebar:
        st.header("Configuration")

        st.session_state.database = st.selectbox(
            "Vector Database",
            ("vectara", "chroma"),
        )

        st.session_state.model_provider = st.selectbox(
            "LLM Provider",
            ("Google Gemini", "TogetherAI"),
        )

        if st.session_state.model_provider == "TogetherAI":
            st.session_state.model = st.selectbox(
                "Model",
                ("Llama-3", "Mistral-7B"),
            )
        else:
            st.session_state.model = "gemini-pro"

    input_fields()

    st.button("Submit Documents", on_click=process_documents)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message("human").write(message[0])
        st.chat_message("ai").write(message[1])

    if query := st.chat_input():
        st.chat_message("human").write(query)
        chain = make_rag_chain(
            st.session_state.vector_db, model=st.session_state.model
        )
        response = ask_question(chain, query)
        st.chat_message("ai").write(response)


if __name__ == "__main__":
    main_streamlit()
