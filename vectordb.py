import chromadb
from typing import List

from chromadb.utils import embedding_functions

# By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings.
default_ef = embedding_functions.DefaultEmbeddingFunction()


# about embedding function: https://docs.trychroma.com/embeddings

class SimilarProductVectorDB:
    def __init__(self, collection_name, embedding_function=default_ef, distance_function="cosine", n_query_result=10):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        # create collection
        chroma_client = chromadb.Client()
        self.collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": distance_function}
        )
        self.n_query_result = n_query_result

    def add_documents(self, documents: List[str], metadatas: List[dict], product_ids: List[str]):
        for _documents, _metadatas, _product_ids in zip(chunk(documents), chunk(metadatas), chunk(product_ids)):
            self.collection.add(
                documents=_documents,
                metadatas=_metadatas,
                ids=_product_ids,
            )

    def add_embeddings(self, embeddings: List[list], metadatas: List[dict], product_ids: List[str]):
        for _embeddings, _metadatas, _product_ids in zip(chunk(embeddings), chunk(metadatas), chunk(product_ids)):
            self.collection.add(
                embeddings=_embeddings,
                metadatas=_metadatas,
                ids=_product_ids,
            )

    def query_with_product_id(self, product_id):
        # get embedding of the product_id
        product_dict = self.collection.get(ids=product_id, include=["embeddings", "metadatas"])
        query_embeddings = product_dict["embeddings"]
        query_class = product_dict["metadatas"][0]["class"]
        # query
        results = self.collection.query(
            query_embeddings=query_embeddings,
            n_results=self.n_query_result,
            where={
                "$and": [
                    {
                        "class": {
                            "$eq": query_class
                        },
                    },
                    {
                        "stock_level": {
                            "$gt": 0
                        }
                    },
                ]
            }  # default filters: same class and in stock
        )
        return results


def chunk(lst, batch_size=100):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]
