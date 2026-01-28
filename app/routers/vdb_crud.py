from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import logging
import os

from app.schemas.vector_storage import (
    SearchRequest,
    SearchResponse,
    SearchResultItem,
    DeleteByMetadataRequest,
)
from app.services.vector_storage import VectorMemory
from app.services.documents_parser import DBNParser
from app.models.parameters import (
    ChunkingParameters,
    BatchWorker,
    SearchParameters,
    BiEncoderParams,
    CrossEncoderParams
)
from app.services.embedders import HFBiEmbedder, HFCrossEncoder

logger = logging.getLogger("VectorMemoryRouter")

router = APIRouter(prefix="/vector-memory", tags=["Vector Memory"])

UPLOAD_DIR = Path("tmp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_index")
logger.info(f"Chroma persist directory: {CHROMA_PERSIST_DIR}")

try:
    vector_memory = VectorMemory(
        bi_embedder=HFBiEmbedder(params=BiEncoderParams()),
        cross_encoder=HFCrossEncoder(params=CrossEncoderParams()),
        persist_path=CHROMA_PERSIST_DIR,
    )
    logger.info("VectorMemory initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize VectorMemory: {str(e)}")
    raise
try:
    parser = DBNParser(ChunkingParameters(), BatchWorker())
    logger.info("DocumentParser initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize DocumentParser: {str(e)}")
    raise


def index_file(path: Path):
    try:
        logger.info(f"Starting indexing: {path}")
        docs = parser.load(str(path), "file")
        if not docs:
            logger.warning(f"No documents parsed from {path}")
            return
        vector_memory.add_documents(docs)
        logger.info(f"Successfully indexed {len(docs)} chunks from {path}")
        try:
            path.unlink()
            logger.info(f"Deleted temporary file: {path}")
        except Exception as e:
            logger.warning(f"Failed to delete temp file: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to index file {path}: {str(e)}")
        raise


@router.post("/documents/file")
async def add_from_file(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None,
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    allowed_extensions = {'.pdf', '.docx', '.txt', '.md', '.html'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
        )
    try:
        path = UPLOAD_DIR / file.filename
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        max_size = 50 * 1024 * 1024
        if len(content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large: {len(content)} bytes (max: {max_size})"
            )
        path.write_bytes(content)
        background_tasks.add_task(index_file, path)
        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes)")
        return {
            "status": "accepted",
            "filename": file.filename,
            "size_bytes": len(content),
            "message": "File is being processed in background"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to upload file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/documents/url")
def add_from_url(url: str):
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="URL is required")
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(
            status_code=400,
            detail="URL must start with http:// or https://"
        )
    try:
        logger.info(f"Loading from URL: {url}")
        docs = parser.load(url, "url")
        if not docs:
            raise HTTPException(status_code=400, detail="No content extracted from URL")
        vector_memory.add_documents(docs)
        logger.info(f"Successfully indexed {len(docs)} chunks from {url}")
        return {
            "status": "ok",
            "source": url,
            "chunks_added": len(docs),
            "message": "URL content indexed successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load URL: {str(e)}")


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest):
    try:
        params = SearchParameters(**request.dict())
        hits = vector_memory.search(params)
        return SearchResponse(
            results=[
                SearchResultItem(
                    score=h.score,
                    content=h.document.page_content,
                    metadata=h.document.metadata,
                )
                for h in hits
            ],
            total=len(hits)
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.delete("/delete")
def delete_by_metadata(request: DeleteByMetadataRequest):
    if not request.filter_metadata:
        raise HTTPException(status_code=400, detail="Filter metadata is required")
    try:
        vector_memory.delete_documents(request.filter_metadata)
        return {
            "status": "ok",
            "filter": request.filter_metadata,
            "message": "Documents deleted successfully"
        }
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.delete("/clear")
def clear():
    try:
        vector_memory.clear()
        return {
            "status": "ok",
            "message": "Vector store cleared successfully"
        }
    except Exception as e:
        logger.error(f"Clear failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")


@router.get("/status")
def status():
    try:
        stats = vector_memory.get_stats()
        return stats

    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e),
            "num_documents": 0
        }