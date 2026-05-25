import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


class EmbeddingService:

    _bi_encoder = None
    _cross_encoder = None

    def __init__(self):

        if EmbeddingService._bi_encoder is None:
            EmbeddingService._bi_encoder = SentenceTransformer("all-mpnet-base-v2")

        if EmbeddingService._cross_encoder is None:
            EmbeddingService._cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.bi_encoder = EmbeddingService._bi_encoder
        self.cross_encoder = EmbeddingService._cross_encoder

    def encode(self, text: str):
        return self.bi_encoder.encode(text)

    def cosine_similarity(self, vec1, vec2):
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1) * np.linalg.norm(vec2)
        )

    def cross_score(self, pairs):
        return self.cross_encoder.predict(pairs)