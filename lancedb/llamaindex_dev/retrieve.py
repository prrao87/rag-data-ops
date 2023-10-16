import os
import shutil

import lancedb
from lancedb.pydantic import LanceModel, Vector, pydantic_to_schema
from llama_index import SimpleDirectoryReader, ServiceContext, StorageContext
# from llama_index.llms.mock import MockLLM
from llama_index.indices.vector_store import VectorStoreIndex
from llama_index.vector_stores import LanceDBVectorStore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index import get_response_synthesizer, set_global_service_context


# LanceDB pydantic schema
class Content(LanceModel):
    text: str
    vector: Vector(384)


def create_lance_table(table_name: str) -> lancedb.table.LanceTable:
    try:
        # Create empty table if it does not exist
        tbl = db.create_table(table_name, schema=pydantic_to_schema(Content), mode="overwrite")
    except OSError:
        # If table exists, open it
        tbl = db.open_table(table_name, mode="append")
    return tbl


def main():
    documents = SimpleDirectoryReader("../../data").load_data()
    print("Document ID:", documents[0].doc_id, "Document Hash:", documents[0].hash)

    TABLE_NAME = "countries"
    vector_store = LanceDBVectorStore(
        uri="./db",
        table_name = TABLE_NAME,
    )
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
    )
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    return index


if __name__ == "__main__":
    DB_NAME = "./db"
    TABLE = "countries"
    if os.path.exists(DB_NAME):
        # Clear DB if it exists
        shutil.rmtree(DB_NAME)

    MODEL = "local:sentence-transformers/all-MiniLM-L6-v2"
    service_context = ServiceContext.from_defaults(llm=None, embed_model=MODEL)
    set_global_service_context(service_context)

    db = lancedb.connect(DB_NAME)
    print("Finished loading documents to LanceDB")

    index = main()

    # Configure retriever
    retriever = index.as_retriever(retriever_mode="default")
    # # Configure response synthesizer
    # response_synthesizer = get_response_synthesizer()

    # # Assemble query engine
    # query_engine = RetrieverQueryEngine(
    #     retriever=retriever,
    #     response_synthesizer=response_synthesizer,
    #     node_postprocessors=[
    #         SimilarityPostprocessor(similarity_cutoff=0.8)
    #     ]
    # )
    query_engine = RetrieverQueryEngine.from_args(retriever, response_mode='compact')

    query = "Is Tonga a monarchy or a democracy"
    response = query_engine.query(query)
    print(response)
