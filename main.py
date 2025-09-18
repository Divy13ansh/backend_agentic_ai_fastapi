from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Annotated
from services import process_query, list_datasets
from dynamic_data_ingestion import ingest_data
from services import decision_maker

app = FastAPI(title="CFOx.ai Simple")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    dataset_name: Optional[str] = "my_json_collection"

@app.post("/query")
async def query_endpoint(request: QueryRequest):
    return process_query(request.query, request.dataset_name)

@app.get("/datasets")
async def get_datasets():
    return list_datasets()

@app.post('/dataset')
async def add_dataset(file: Annotated[UploadFile, File()]):
    filename = file.filename
    content_type = file.content_type
    contents = await file.read()
    ingest_data(contents, filename)
    return {"message": f"Dataset '{filename}' ingested successfully."}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "CFOx.ai Simple RAG API",
        "endpoints": {
            "/query": "POST - Query with dataset",
            "/datasets": "GET - List all datasets",
            "/health": "GET - Health check"
        }
    }

@app.post("/decision_maker")
async def decision_maker_endpoint(request: QueryRequest):
    return decision_maker(request.query, request.dataset_name)



# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
