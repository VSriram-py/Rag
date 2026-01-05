# Compatibility shim: expose the functions from the top-level rag.py
# This helps test runners that import the module as a package named `rag`.
import importlib.util
import os

_here = os.path.dirname(os.path.dirname(__file__))
_rag_path = os.path.join(_here, 'rag.py')

if os.path.exists(_rag_path):
    spec = importlib.util.spec_from_file_location('rag_py', _rag_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Export the commonly-used names
    for name in ('get_embedding', 'cross_encoder_score', 'get_conn', 'search_with_reranking', 'EMBEDDING_DIM'):
        if hasattr(mod, name):
            globals()[name] = getattr(mod, name)
else:
    # Fallback: empty module
    pass
