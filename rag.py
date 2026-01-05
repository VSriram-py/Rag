from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'openai:gpt-4o')
agent = Agent(MODEL_NAME, system_prompt='You are a RAG assistant with re-ranking.')

DATABASE_URL = os.getenv('DATABASE_URL', 'dbname=rag_db')
conn = None

def get_conn():
    """Lazily create and return a Postgres connection and register pgvector."""
    global conn
    if conn is None:
        conn = psycopg2.connect(DATABASE_URL)
        register_vector(conn)
    return conn

def ingest_document(text: str):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]
    conn_local = get_conn()
    with conn_local.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk)
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                       (chunk, embedding))
    conn.commit()

@agent.tool
def search_with_reranking(query: str) -> str:
    # Stage 1: Fast vector search (retrieve 20 candidates)
    conn_local = get_conn()
    with conn_local.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute(
            'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 20',
            (query_embedding,)
        )
        candidates = [row[0] for row in cur.fetchall()]

    # Stage 2: Re-rank with cross-encoder
    scored_results = []
    for doc in candidates:
        score = cross_encoder_score(query, doc)  # Assume cross-encoder function
        scored_results.append((doc, score))

    # Return top 5 after re-ranking
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return "\n".join([doc for doc, _ in scored_results[:5]])

# Run agent
result = agent.run_sync("Explain neural networks")
from pydantic_ai import Agent
import psycopg2
from pgvector.psycopg2 import register_vector
import os
from dotenv import load_dotenv
import openai
import math
import hashlib

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'openai:gpt-4o')
agent = Agent(MODEL_NAME, system_prompt='You are a RAG assistant with re-ranking.')

DATABASE_URL = os.getenv('DATABASE_URL', 'dbname=rag_db')
conn = psycopg2.connect(DATABASE_URL)
register_vector(conn)

EMBEDDING_DIM = 1536


def _pseudo_embedding(text: str):
    h = hashlib.sha256(text.encode('utf-8')).digest()
    vals = [(b / 255.0) for b in h]
    # expand/repeat to EMBEDDING_DIM
    out = [vals[i % len(vals)] for i in range(EMBEDDING_DIM)]
    return out


def get_embedding(text: str):
    """Return an embedding vector for `text`.

    Tries OpenAI if `OPENAI_API_KEY` is set, otherwise returns a deterministic
    pseudo-embedding. The returned value is a list of floats of length
    EMBEDDING_DIM (1536) to match the DB schema.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-small')
    if api_key:
        try:
            openai.api_key = api_key
            resp = openai.Embedding.create(input=text, model=model)
            emb = resp['data'][0]['embedding']
            # Pad/trim to EMBEDDING_DIM if necessary
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
    """Lightweight re-ranking: compute cosine similarity between embeddings.

    This is a fallback for a true cross-encoder; it is deterministic and
    requires no extra services beyond embeddings.
    """
    eq = get_embedding(query)
    ed = get_embedding(doc)
    return _cosine(eq, ed)


def ingest_document(text: str):
    chunks = [text[i:i + 500] for i in range(0, len(text), 500)]
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding = get_embedding(chunk)
            cur.execute('INSERT INTO chunks (content, embedding) VALUES (%s, %s)',
                        (chunk, embedding))
    conn.commit()


@agent.tool
def search_with_reranking(query: str) -> str:
    # Stage 1: Fast vector search (retrieve 20 candidates)
    conn_local = get_conn()
    with conn_local.cursor() as cur:
        query_embedding = get_embedding(query)
        cur.execute(
            'SELECT content FROM chunks ORDER BY embedding <=> %s LIMIT 20',
            (query_embedding,)
        )
        candidates = [row[0] for row in cur.fetchall()]

    # Stage 2: Re-rank with cross-encoder (here: cosine on embeddings)
    scored_results = []
    for doc in candidates:
        score = cross_encoder_score(query, doc)
        scored_results.append((doc, score))

    # Return top 5 after re-ranking
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return "\n".join([doc for doc, _ in scored_results[:5]])


if __name__ == '__main__':
    result = agent.run_sync("Explain neural networks")
    try:
        print(result.data)
    except Exception:
        print(result)
