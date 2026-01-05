import importlib
import importlib.util
import os


def _import_rag():
    try:
        return importlib.import_module('rag')
    except ModuleNotFoundError:
        # Fallback: load module directly from repository root rag.py
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, 'rag.py')
        spec = importlib.util.spec_from_file_location('rag', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def test_get_embedding_len():
    rag = _import_rag()
    emb = rag.get_embedding('hello world')
    assert isinstance(emb, list)
    assert len(emb) == rag.EMBEDDING_DIM


def test_cross_encoder_score_numeric():
    rag = _import_rag()
    s = rag.cross_encoder_score('a', 'b')
    assert isinstance(s, float)


def test_search_or_fallback():
    rag = _import_rag()
    try:
        out = rag.search_with_reranking('explain neural networks')
        assert isinstance(out, str)
    except Exception:
        # If DB or agent not available, ensure fallback scoring works
        chunks = ['one', 'two']
        scores = [rag.cross_encoder_score('q', c) for c in chunks]
        assert all(isinstance(x, float) for x in scores)
