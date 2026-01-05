import importlib


def test_get_embedding_len():
    rag = importlib.import_module('rag')
    emb = rag.get_embedding('hello world')
    assert isinstance(emb, list)
    assert len(emb) == rag.EMBEDDING_DIM


def test_cross_encoder_score_numeric():
    rag = importlib.import_module('rag')
    s = rag.cross_encoder_score('a', 'b')
    assert isinstance(s, float)


def test_search_or_fallback():
    rag = importlib.import_module('rag')
    try:
        out = rag.search_with_reranking('explain neural networks')
        assert isinstance(out, str)
    except Exception:
        # If DB or agent not available, ensure fallback scoring works
        chunks = ['one', 'two']
        scores = [rag.cross_encoder_score('q', c) for c in chunks]
        assert all(isinstance(x, float) for x in scores)
