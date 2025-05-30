# oraculo_api.py  – Célula Oráculo v2.5 (Railway Free)
"""
Função  ▸ recebe uma pergunta ▶ gera embedding ▶ consulta Pinecone ▶ devolve matches
Segurança ▸ não armazena chave no código; usa variáveis de ambiente.
Estabilidade ▸ sem estado entre requisições, CPU-only, memória < 200 MB.
"""

import os, asyncio
from functools import lru_cache
from typing import List

import pinecone
from fastapi import FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# ───────────────────────── Variáveis de Ambiente ──────────────────────────
API_KEY  = os.getenv("PINECONE_API_KEY")            # 🔑 sua chave real
ENV      = os.getenv("PINECONE_ENVIRONMENT")        # ex: us-east-1
INDEX    = os.getenv("PINECONE_INDEX")              # ex: vazium
NS       = os.getenv("PINECONE_NAMESPACE") or None  # opcional
MODEL_ID = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K_MAX = int(os.getenv("TOP_K_MAX", "20"))

if not all([API_KEY, ENV, INDEX]):
    raise RuntimeError("Defina PINECONE_API_KEY, PINECONE_ENVIRONMENT e PINECONE_INDEX.")

# ──────────────────────────── Pinecone Client ─────────────────────────────
pinecone.init(api_key=API_KEY, environment=ENV, connection_timeout=5)
index = pinecone.Index(INDEX)
DIM   = index.describe_index_stats().get("dimension", 384)   # segurança contra dim errada

# ─────────────────────── Sentence-Transformer (singleton) ──────────────────
@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_ID)
    if model.get_sentence_embedding_dimension() != DIM:
        raise RuntimeError(f"Modelo {MODEL_ID} dim {model.get_sentence_embedding_dimension()} ≠ {DIM}.")
    model.eval()
    return model

@lru_cache(maxsize=64)
def encode(text: str) -> List[float]:
    """Cacheia embeddings de até 64 perguntas repetidas."""
    return _embedder().encode(text, normalize_embeddings=True).tolist()

# ───────────────────────────── FastAPI App ────────────────────────────────
app = FastAPI(title="Célula Oráculo v2.5")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Consulta(BaseModel):
    pergunta: str = Field(min_length=3, description="Pergunta em PT-BR ou EN")
    top_k: int    = Field(default=5, le=TOP_K_MAX, description="Máximo de vetores a retornar")

@app.get("/healthz"), app.head("/healthz")
def health() -> dict:
    return {"status": "ok", "dim": DIM}

@app.post("/consultar", status_code=status.HTTP_200_OK)
async def consultar(c: Consulta):
    """Endpoint principal; retorna lista de matches ou 204 se nenhum."""
    vector = encode(c.pergunta.strip())

    try:
        res = await asyncio.to_thread(
            index.query,
            vector=vector,
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

# Execução local opcional:  python oraculo_api.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("oraculo_api:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
