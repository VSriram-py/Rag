#!/usr/bin/env python3
"""Test runner for the RAG demo.

Ingests a sample document and runs `search_with_reranking()`.
If Postgres is unavailable, falls back to an in-memory test using the
fallback embedding and reranker in `rag.py`.
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # dotenv is optional for tests and local runs
    pass

SAMPLE_TEXT = (
    "Neural networks are computing systems inspired by biological neural networks. "
    "They learn from examples and are widely used in tasks like image recognition, "
    "language modeling, and reinforcement learning."
)


def main():
    try:
        # Importing rag now is safe because rag uses a lazy DB connection.
        import rag

        # Ensure table exists (idempotent)
        try:
            conn = rag.get_conn()
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS chunks (
                      id SERIAL PRIMARY KEY,
                      content TEXT NOT NULL,
                      embedding vector(1536)
                    );
                    """
                )
                conn.commit()
        except Exception as e:
            print("Warning: could not prepare DB schema (continuing):", e)

        print("Ingesting sample document...")
        try:
            rag.ingest_document(SAMPLE_TEXT)
        except Exception as e:
            print("Warning: ingest_document failed (continuing):", e)

        print("Running search_with_reranking(...)...")
        try:
            out = rag.search_with_reranking("explain neural networks")
            print("Result:\n", out)
        except Exception as e:
            print("Warning: search_with_reranking failed:", e)
            # Fallback: run an in-memory rerank
            chunks = [SAMPLE_TEXT[i:i+500] for i in range(0, len(SAMPLE_TEXT), 500)]
            scores = [(c, rag.cross_encoder_score("explain neural networks", c)) for c in chunks]
            scores.sort(key=lambda x: x[1], reverse=True)
            print("Fallback top result:\n", scores[0][0])

    except Exception as e:
        print("Unable to import `rag` module (DB may be unavailable). Running local dry-run.")
        # Local dry-run using rag's fallback implementations if possible
        try:
            import importlib
            rag = importlib.import_module('rag')
            chunks = [SAMPLE_TEXT[i:i+500] for i in range(0, len(SAMPLE_TEXT), 500)]
            scores = [(c, rag.cross_encoder_score("explain neural networks", c)) for c in chunks]
            scores.sort(key=lambda x: x[1], reverse=True)
            print("Dry-run top result:\n", scores[0][0])
        except Exception as e2:
            print("Dry-run also failed:", e2)


if __name__ == '__main__':
    main()
