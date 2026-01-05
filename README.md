# Rag

A small Retrieval-Augmented Generation (RAG) example using `pydantic_ai`, Postgres with `pgvector`, and an OpenAI model. This repository contains a minimal demo in `rag.py` that shows ingestion and a reranking search tool.

**Quick Setup**

- **Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- **Environment**

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY and DATABASE_URL
```

- **Database**

This project expects a Postgres database with the `pgvector` extension enabled and a `chunks` table. Example SQL:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(1536)
);
```

A ready-to-run migration is provided at `sql/create_chunks_table.sql`. To apply it using `psql`:

```bash
# from your project root, with DATABASE_URL set (example in .env)
psql "$DATABASE_URL" -f sql/create_chunks_table.sql
```

- **Run**

```bash
python rag.py
```

Notes:
- `rag.py` uses placeholder functions `get_embedding()` and `cross_encoder_score()` which need implementations connected to your embedding and cross-encoder services. The repository now provides simple fallback implementations.
- The `Agent` is configured via `pydantic_ai` in `rag.py`; set model via `MODEL_NAME` or directly in code.

If you want help wiring up embeddings or a cross-encoder, tell me which provider you prefer and I can add example implementations.
