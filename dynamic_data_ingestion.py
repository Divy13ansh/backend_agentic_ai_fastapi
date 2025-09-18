import json
import re
import io
from config import *
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import fitz  # PyMuPDF
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import uuid

# Load the desired MiniLM model
model = SentenceTransformer('all-MiniLM-L6-v2')


client = QdrantClient(url = QDRANT_URL, api_key = QDRANT_API_KEY)  # or your cloud endpoint

def create_collection_if_not_exists(name: str, vector_size: int):
    existing = [c.name for c in client.get_collections().collections]

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        # print(f"✅ Created collection: {name}")
    else:
        # print(f"⚡ Collection {name} already exists")
        pass

def get_embedding(text):
    embedding = model.encode(text)  # returns numpy array
    return embedding.tolist()                 # convert to list if storing in Qdrant


def ingest_data(content, filename):
    doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append({"id": str(uuid.uuid4()), "page_num": i+1, "content": text})

    for page in pages:
        page["embedding"] = get_embedding(page["content"])
    
    create_collection_if_not_exists(filename, vector_size=len(page["embedding"]))

    client.upsert(
        collection_name=filename,
        points=[
            PointStruct(
                id=page["id"],  # int or UUID string
                vector=page["embedding"],
                payload={
                    "filename": filename,
                    "page_num": page["page_num"],
                    "content": page["content"],
                },
            )
            for page in pages
        ],
    )
    return {"message": f"Data from {filename} ingested successfully."}
    