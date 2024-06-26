import os

from dotenv import load_dotenv
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import Together

load_dotenv()

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

SYS_PROMPT_RAG = """Usted es un asistente médico y su función es responder a la pregunta basándose en el contexto dado, proveniente de la historia clínica de un paciente. Al responder la pregunta, utilice y cite elementos relevantes del contexto para justificar su respuesta. Si la información disponible es insuficiente para formular una respuesta informada, indique claramente que no se dispone de suficiente información para responder. Sea preciso y conciso."""
USER_PROMPT_RAG = """Pregunta: {question}\n\n\nContexto:\n\n{context}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content.replace("\n\n", "\n") for doc in docs)


def get_question(input):
    if not input:
        return None
    elif isinstance(input, str):
        return input
    elif isinstance(input, dict) and "question" in input:
        return input["question"]
    elif isinstance(input, BaseMessage):
        return input.content
    else:
        raise Exception(
            "string or dict with 'question' key expected as RAG chain input."
        )


def ask_question(chain, query):
    response = chain.invoke({"question": query})
    return response


def build_rag_chat_prompt(
    system_prompt=SYS_PROMPT_RAG, user_prompt=USER_PROMPT_RAG
):
    messages = [
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(user_prompt),
    ]

    template_prompt = ChatPromptTemplate.from_messages(messages=messages)

    return template_prompt


def build_rag_prompt(
    system_prompt=SYS_PROMPT_RAG, user_prompt=USER_PROMPT_RAG
):

    template = f"""{system_prompt}\n\n\n{user_prompt}\n\n\nRespuesta: """

    template_prompt = PromptTemplate.from_template(template)

    return template_prompt


def make_rag_chain(vector_db, model="gemini-pro", rag_prompt=None):

    if model == "gemini-pro":
        model = ChatGoogleGenerativeAI(model=model)
        rag_prompt = build_rag_chat_prompt()
    else:
        if model == "Llama-3":
            model_name = "meta-llama/Llama-3-70b-chat-hf"
        else:
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        model = Together(
            model=model_name,
            temperature=0.0,
            max_tokens=512,
            together_api_key=TOGETHER_API_KEY,
        )
        rag_prompt = build_rag_prompt()

    retriever = vector_db.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs,
            "question": RunnableLambda(get_question),
        }
        | rag_prompt
        | model
        | StrOutputParser()
    )

    return rag_chain
