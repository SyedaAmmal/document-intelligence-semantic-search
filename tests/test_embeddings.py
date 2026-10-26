"""Offline tests for the injected Sentence Transformers encoder."""

import unittest
from unittest.mock import patch

import numpy as np

from src.document_intelligence.embeddings import (
    DEFAULT_MODEL_NAME,
    EmbeddingEncoder,
)
from src.document_intelligence.models import DocumentChunk


class FakeEmbeddingModel:
    def __init__(self, output: object | None = None) -> None:
        self.output = output
        self.calls: list[tuple[list[str], dict[str, bool]]] = []

    def encode(
        self,
        sentences: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
    ) -> object:
        self.calls.append(
            (
                sentences,
                {
                    "convert_to_numpy": convert_to_numpy,
                    "normalize_embeddings": normalize_embeddings,
                },
            )
        )
        if self.output is not None:
            return self.output
        return np.array([[3.0, 4.0] for _ in sentences], dtype=np.float32)


class EmbeddingEncoderTests(unittest.TestCase):
    def test_encodes_and_normalizes_text_using_injected_model(self) -> None:
        model = FakeEmbeddingModel()
        encoder = EmbeddingEncoder(model=model)

        embeddings = encoder.encode_texts(["document", "query"])

        np.testing.assert_allclose(embeddings, [[0.6, 0.8], [0.6, 0.8]])
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertEqual(
            model.calls,
            [
                (
                    ["document", "query"],
                    {
                        "convert_to_numpy": True,
                        "normalize_embeddings": True,
                    },
                )
            ],
        )

    def test_chunk_and_query_encoding_share_the_same_model(self) -> None:
        model = FakeEmbeddingModel()
        encoder = EmbeddingEncoder(model=model)
        chunk = DocumentChunk(
            chunk_id="chunk-1",
            text="chunk text",
            source_filename="notes.txt",
            source_page_numbers=(),
        )

        chunk_vector = encoder.encode_chunks([chunk])
        query_vector = encoder.encode_query("search text")

        np.testing.assert_allclose(chunk_vector[0], query_vector)
        self.assertEqual(
            [call[0] for call in model.calls],
            [["chunk text"], ["search text"]],
        )

    def test_empty_input_does_not_load_model(self) -> None:
        encoder = EmbeddingEncoder()

        embeddings = encoder.encode_texts([])

        self.assertEqual(embeddings.shape, (0, 0))
        self.assertIsNone(encoder._model)

    def test_default_model_loads_only_on_first_nonempty_encoding(self) -> None:
        fake_model = FakeEmbeddingModel()
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=fake_model,
        ) as model_constructor:
            encoder = EmbeddingEncoder()
            self.assertEqual(encoder.model_name, DEFAULT_MODEL_NAME)
            model_constructor.assert_not_called()

            encoder.encode_texts(["first request"])
            encoder.encode_texts(["second request"])

        model_constructor.assert_called_once_with(DEFAULT_MODEL_NAME)
        self.assertEqual(len(fake_model.calls), 2)

    def test_rejects_empty_inputs_and_invalid_model_output(self) -> None:
        encoder = EmbeddingEncoder(model=FakeEmbeddingModel())
        with self.assertRaises(ValueError):
            encoder.encode_texts([" \t"])
        with self.assertRaises(ValueError):
            encoder.encode_query(" ")
        with self.assertRaises(TypeError):
            encoder.encode_texts("not a list")  # type: ignore[arg-type]

        bad_outputs = (
            np.array([1.0, 2.0]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[0.0, 0.0]]),
            np.array([[np.nan, 1.0]]),
        )
        for output in bad_outputs:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    EmbeddingEncoder(model=FakeEmbeddingModel(output)).encode_texts(
                        ["input"]
                    )


if __name__ == "__main__":
    unittest.main()
