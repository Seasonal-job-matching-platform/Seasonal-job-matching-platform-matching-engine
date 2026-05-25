import numpy as np
from app.embedding_service import EmbeddingService
from app.feature_engineering import build_feature_vector


class TalentRanker:

    def __init__(self):
        self.embedder = EmbeddingService()
        self.cached_candidate_embeddings = {}

    def build_candidate_cache(self, candidates_df):

        for _, row in candidates_df.iterrows():

            resume_text = " ".join([
                " ".join(row["skills"] or []),
                " ".join(row["experience"] or []),
                " ".join(row["education"] or []),
                " ".join(row["certificates"] or [])
            ])

            embedding = self.embedder.encode(resume_text)

            self.cached_candidate_embeddings[row["user_id"]] = embedding

    def rank(self, job: dict, candidates_df, top_k: int = 5):

        # Build cache once
        if not self.cached_candidate_embeddings:
            self.build_candidate_cache(candidates_df)

        job_text = " ".join([
            job.get("title", ""),
            job.get("description", ""),
            " ".join(job.get("requirements") or []),
            " ".join(job.get("categories") or [])
        ])

        job_embedding = self.embedder.encode(job_text)

        stage1_results = []

        # ---------------------------
        # Stage 1: Fast Retrieval
        # ---------------------------
        for _, row in candidates_df.iterrows():

            candidate_embedding = self.cached_candidate_embeddings.get(row["user_id"])

            semantic_sim = self.embedder.cosine_similarity(
                job_embedding, candidate_embedding
            )

            stage1_results.append((row, semantic_sim))

        # Top 20 retrieval
        stage1_results.sort(key=lambda x: x[1], reverse=True)
        top_candidates = stage1_results[:20]

        # ---------------------------
        # Stage 2: CrossEncoder Re-ranking
        # ---------------------------
        pairs = []
        rows = []

        for row, _ in top_candidates:

            candidate_text = " ".join([
                " ".join(row["skills"] or []),
                " ".join(row["experience"] or []),
                " ".join(row["education"] or []),
                " ".join(row["certificates"] or [])
            ])

            pairs.append([job_text, candidate_text])
            rows.append(row)

        cross_scores = self.embedder.cross_score(pairs)

        results = []

        for row, cross_score in zip(rows, cross_scores):

            candidate = row.to_dict()
            candidate["score"] = float(cross_score)
            results.append(candidate)

        results.sort(key=lambda x: x["score"], reverse=True)

# Remove score before returning
        final_results = []
        for candidate in results[:top_k]:
            candidate_copy = candidate.copy()
            candidate_copy.pop("score", None)
            final_results.append(candidate_copy)
            
        return final_results