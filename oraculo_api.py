# oraculo_api.py  – Célula Oráculo v2.4
import os, asyncio
from functools import lru_cache
from typing import List

import pinecone
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ── variáveis obrigatórias ────────────────────────────────────────────────
API_KEY  = os.getenv("PINECONE_API_KEY")
ENV      = os.getenv("PINECONE_ENVIRONMENT")          # ex: us-east-1
INDEX    = os.getenv("PINECONE_INDEX")                # ex: vazium
NS       = os.getenv("PINECONE_NAMESPACE") or None    # opcional
MODEL_ID = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "20"))
if not all([API_KEY, ENV, INDEX]):
    raise RuntimeError("🔑  PINECONE_API_KEY / ENV / INDEX não definidos.")

# ── Pinecone + embedder ───────────────────────────────────────────────────
pinecone.init(api_key=API_KEY, environment=ENV, connection_timeout=5)
index = pinecone.Index(INDEX)
DIM   = index.describe_index_stats().get("dimension", 384)

@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    m = SentenceTransformer(MODEL_ID)          # carrega 1× só
    if m.get_sentence_embedding_dimension() != DIM:
        raise RuntimeError(f"Dimensão {MODEL_ID} ≠ {DIM} do índice.")
    m.eval()
    return m

@lru_cache(maxsize=32)
def encode(text: str) -> List[float]:
    return embedder().encode(text, normalize_embeddings=True).tolist()

# ── FastAPI ───────────────────────────────────────────────────────────────
app = FastAPI(title="Célula Oráculo v2.4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Consulta(BaseModel):
    pergunta: str = Field(min_length=3)
    top_k: int    = Field(default=5, le=TOP_K_MAX)

@app.get("/healthz"), app.head("/healthz")
def health():                                    # ping rápido
    return {"status": "ok", "dim": DIM}

@app.post("/consultar", status_code=status.HTTP_200_OK)
async def consultar(c: Consulta):
    vec = encode(c.pergunta.strip())
    try:
        res = await asyncio.to_thread(
            index.query,
            vector=vec,
            top_k=c.top_k,
            include_metadata=True,
            namespace=NS
        )
    except Exception as e:
        raise HTTPException(503, detail=f"Pinecone indisponível: {e}")

    matches = res.get("matches", [])
    if not matches:
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return [
        {
            "id": m["id"],
            "score": round(m["score"], 4),
            "titulo": m["metadata"].get("titulo", ""),
            "sementes": m["metadata"].get("sementes_ativadas", [])
        } for m in matches
    ]

if __name__ == "__main__":                       # uso local opcional
    import uvicorn, os
    uvicorn.run("oraculo_api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
