-- SQL migration: create_chunks_table.sql
-- Enables the pgvector extension and creates the `chunks` table used by rag.py

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding vector(1536)
);
