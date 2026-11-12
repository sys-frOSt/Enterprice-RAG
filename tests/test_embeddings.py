import pytest

from app.embeddings.embedding_model import BGEEmbeddings


def test_embedding_dimension_validation_accepts_bge_m3_shape() -> None:
    embeddings = object.__new__(BGEEmbeddings)

    embeddings._validate([[0.0] * 1024])


def test_embedding_dimension_validation_rejects_wrong_shape() -> None:
    embeddings = object.__new__(BGEEmbeddings)

    with pytest.raises(ValueError, match="1024"):
        embeddings._validate([[0.0] * 3])
