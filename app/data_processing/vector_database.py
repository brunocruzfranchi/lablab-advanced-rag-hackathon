import json
import os
import shutil
from time import sleep
from typing import List

import requests
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_community.vectorstores import Chroma, Vectara

EMBED_DELAY = 0.02  # 20 milliseconds

VECTARA_CUSTOMER_ID = os.getenv("VECTARA_CUSTOMER_ID")
VECTARA_CORPUS_ID = os.getenv("VECTARA_CORPUS_ID")
VECTARA_API_KEY = os.getenv("VECTARA_API_KEY")
VECTARA_AUTH_TOKEN = os.getenv("VECTARA_AUTH_TOKEN")


class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


def reset_vectara_corpus(auth_api_key: str, customer_id: int, corpus_id: int):

    url = "https://api.vectara.io/v1/reset-corpus"

    payload = json.dumps({"corpusId": corpus_id})

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "customer-id": f"{customer_id}",
        "x-api-key": f"{auth_api_key}",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response


def create_vectara_db(texts):

    vector_db = Vectara(
        vectara_customer_id=VECTARA_CUSTOMER_ID,
        vectara_corpus_id=VECTARA_CORPUS_ID,
        vectara_api_key=VECTARA_API_KEY,
    )

    reset_vectara_corpus(
        VECTARA_AUTH_TOKEN, VECTARA_CUSTOMER_ID, VECTARA_CORPUS_ID
    )

    vector_db.from_documents(texts, FakeEmbeddings(size=768))

    return vector_db


def create_chroma_db(texts, collection_name="chroma"):

    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="hackathon-pln-es/paraphrase-spanish-distilroberta"
    )

    proxy_embeddings = EmbeddingProxy(embeddings)

    # Find and delete the 'store' folder
    store_folder = os.path.join(os.getcwd(), "store")

    if os.path.exists(store_folder):
        shutil.rmtree(store_folder)

    db = Chroma.from_documents(
        documents=texts,
        embedding=proxy_embeddings,
        persist_directory=os.path.join("store/", collection_name),
    )

    db.persist()

    return db


def create_vector_db(texts, vector_db="chroma"):

    if vector_db == "chroma":
        db = create_chroma_db(texts)
    elif vector_db == "vectara":
        db = create_vectara_db(texts)

    return db
