from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pathlib import Path
from typing import Dict, Any

from app.schemas.vector_storage import SearchResultItem, SearchResponse, SearchRequest
from app.services.vector_storage import VectorMemory
from app.services.documents_parser import DBNParser
from app.models.parameters import ChunkingParameters, SearchParameters, BatchWorker

router = APIRouter(
    prefix="/vector-memory",
    tags=["Vector Memory"],
)

vector_memory: VectorMemory = None
parser = DBNParser(ChunkingParameters(), BatchWorker())

UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/documents/url")
def add_from_url(url: str):
    try:
        docs = parser.load(url, "url")
        vector_memory.add_documents(docs)
        return {
            "status": "ok",
            "source": url,
            "chunks_added": len(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/documents/file")
def add_from_file(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        file_path.write_bytes(file.file.read())
        docs = parser.load(str(file_path), "file")
        vector_memory.add_documents(docs)
        return {
            "status": "ok",
            "filename": file.filename,
            "chunks_added": len(docs),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/search", response_model=SearchResponse)
def search_endpoint(request: SearchRequest):

    params = SearchParameters(
        query=request.query,
        top_k_retrieve=request.top_k_retrieve,
        use_reranking=request.use_reranking,
        top_k_reranking=request.top_k_reranking,
        rerank_threshold=request.rerank_threshold,
    )

    results = vector_memory.search(params)

    response_items = []
    for item in results:
        if isinstance(item, tuple):
            score, doc = item
        else:
            score = None
            doc = item
        response_items.append(
            SearchResultItem(
                score=score,
                content=doc.page_content,
                metadata=doc.metadata,
            )
        )

    return SearchResponse(results=response_items)


@router.delete("/documents")
def delete_by_metadata(filter_metadata: Dict[str, Any]):
    try:
        vector_memory.delete_documents(filter_metadata)
        return {"status": "ok", "deleted_by_filter": filter_metadata}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/clear")
def clear_vector_store():
    try:
        vector_memory.clear()
        return {"status": "ok", "message": "Vector store cleared"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
def get_status():
    try:
        stats = vector_memory.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
