import os
import shutil
from time import sleep
from typing import List

from langchain_community.vectorstores import Chroma

EMBED_DELAY = 0.02  # 20 milliseconds


class EmbeddingProxy:
    def __init__(self, embedding):
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


def create_vector_db(texts, embeddings=None, collection_name="chroma"):

    if not embeddings:

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
