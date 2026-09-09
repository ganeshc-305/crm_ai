import os
import tempfile
from src.rag.loaders import load_documents_from_dir, split_documents_to_texts


def test_load_text_file_and_split():
    with tempfile.TemporaryDirectory() as td:
        fpath = os.path.join(td, 'sample.txt')
        content = 'Hello world. ' * 200  # long text
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)

        docs = load_documents_from_dir(td, recursive=False)
        assert len(docs) == 1
        assert 'sample.txt' in docs[0].metadata.get('source')

        texts, metadatas = split_documents_to_texts(docs, chunk_size=500, chunk_overlap=50)
        assert len(texts) >= 1
        assert len(texts) == len(metadatas)
        assert all('source' in m for m in metadatas)
