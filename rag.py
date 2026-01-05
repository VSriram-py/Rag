import os
import math
import hashlib
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_DIM = 1536


def _pseudo_embedding(text: str):
    h = hashlib.sha256(text.encode('utf-8')).digest()
    vals = [(b / 255.0) for b in h]
    out = [vals[i % len(vals)] for i in range(EMBEDDING_DIM)]
    return out


def get_embedding(text: str):
    """Return an embedding vector for `text`.

    Tries OpenAI if `OPENAI_API_KEY` is set, otherwise returns a deterministic
    pseudo-embedding. The returned value is a list of floats of length
    EMBEDDING_DIM (1536).
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            import openai
            model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
            openai.api_key = api_key
            resp = openai.Embedding.create(input=text, model=model)
            emb = resp['data'][0]['embedding']
            if len(emb) < EMBEDDING_DIM:
                emb = emb + [0.0] * (EMBEDDING_DIM - len(emb))
            return emb[:EMBEDDING_DIM]
        except Exception:
            return _pseudo_embedding(text)
    return _pseudo_embedding(text)


def _cosine(a, b):
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def cross_encoder_score(query: str, doc: str) -> float:
    """Fallback re-ranking: cosine similarity between embeddings."""
    eq = get_embedding(query)
    ed = get_embedding(doc)
    return _cosine(eq, ed)


# Database utilities: lazy and non-failing at import time
_conn = None


def get_conn():
    """Attempt to return a psycopg2 connection or None if unavailable.

    This function will not raise on import; callers should handle None.
    """
    global _conn
    if _conn is not None:
        return _conn
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        return None
    try:
        import psycopg2
        from pgvector.psycopg2 import register_vector
        _conn = psycopg2.connect(db_url)
        register_vector(_conn)
        return _conn
    except Exception:
        _conn = None
        return None


def search_with_reranking(query: str) -> str:
    """Two-stage search: vector retrieval (DB) then re-rank with cosine.

    If the DB is not available this raises an Exception so callers can
    fall back to alternative behavior.
    """
    conn = get_conn()
    if conn is None:
        raise Exception("Database not available")

    with conn.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute(
            'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 20',
            (query_embedding,)
        )
        candidates = [row[0] for row in cur.fetchall()]

    scored_results = [(doc, cross_encoder_score(query, doc)) for doc in candidates]
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return "\n".join([doc for doc, _ in scored_results[:5]])
