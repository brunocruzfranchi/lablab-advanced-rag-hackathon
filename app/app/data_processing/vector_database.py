import os
from time import sleep
from typing import List

from langchain_community.vectorstores import Chroma

EMBED_DELAY = 0.02  # 20 milliseconds


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
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
            model_name="hiiamsid/sentence_similarity_spanish_es"
        )

    proxy_embeddings = EmbeddingProxy(embeddings)

    db = Chroma(
        collection_name=collection_name,
        embedding_function=proxy_embeddings,
        persist_directory=os.path.join("store/", collection_name),
    )

    db.add_documents(texts)

    return db
