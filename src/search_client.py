"""
Multi-Modal Video Search Client

Performs fusion search across visual, audio, and transcription
embeddings stored in MongoDB Atlas.

Supports multiple fusion methods:
1. Reciprocal Rank Fusion (RRF) - rank-based fusion, more robust
2. Weighted Score Fusion - simple weighted sum
3. Dynamic Intent Routing - auto-calculates weights based on query intent

RRF formula: score(d) = Σ w_m / (k + rank_m(d))
"""

import math
from typing import Optional
from pymongo import MongoClient

from bedrock_client import BedrockMarengoClient


# Anchor text prompts for dynamic intent routing (Section 4.3 of whitepaper)
ANCHOR_PROMPTS = {
    "visual": "What appears on screen: people, objects, scenes, actions, clothing, colors, and visual composition of the video.",
    "audio": "The non-speech audio in the video: music, sound effects, ambient sound, and other audio elements.",
    "transcription": "The spoken words in the video: dialogue, narration, speech, and what people say."
}


def cosine_similarity(vec1: list, vec2: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


def softmax_with_temperature(scores: dict, temperature: float = 10.0) -> dict:
    """Apply softmax with temperature to get weights."""
    # Scale by temperature
    scaled = {k: v * temperature for k, v in scores.items()}
    # Compute max for numerical stability
    max_val = max(scaled.values())
    # Compute exp
    exp_scores = {k: math.exp(v - max_val) for k, v in scaled.items()}
    # Normalize
    total = sum(exp_scores.values())
    return {k: v / total for k, v in exp_scores.items()}


class VideoSearchClient:
    """Client for multi-modal video search with fusion."""

    # TwelveLabs-style weights: heavily favor visual for most queries
    DEFAULT_WEIGHTS = {
        "visual": 0.8,
        "audio": 0.1,
        "transcription": 0.05
    }

    # RRF constant (standard value used by Elasticsearch, etc.)
    RRF_K = 60

    # Temperature for softmax in dynamic routing
    SOFTMAX_TEMPERATURE = 10.0

    def __init__(
        self,
        mongodb_uri: str,
        database_name: str = "video_search",
        collection_name: str = "video_embeddings",
        index_name: str = "vector_index",
        bedrock_region: str = "us-east-1"
    ):
        self.mongo_client = MongoClient(mongodb_uri)
        self.collection = self.mongo_client[database_name][collection_name]
        self.index_name = index_name
        self.bedrock = BedrockMarengoClient(
            region=bedrock_region,
            output_bucket="tl-brice-media"
        )
        # Anchor embeddings for dynamic routing (lazy initialized)
        self._anchor_embeddings = None

    def initialize_anchors(self) -> dict:
        """
        Pre-compute anchor embeddings for dynamic intent routing.
        Call this at app startup to avoid latency on first query.
        """
        if self._anchor_embeddings is not None:
            return self._anchor_embeddings

        self._anchor_embeddings = {}
        for modality, prompt in ANCHOR_PROMPTS.items():
            result = self.bedrock.get_text_query_embedding(prompt)
            self._anchor_embeddings[modality] = result["embedding"]

        return self._anchor_embeddings

    def get_anchor_embeddings(self) -> dict:
        """Get anchor embeddings, initializing if needed."""
        if self._anchor_embeddings is None:
            self.initialize_anchors()
        return self._anchor_embeddings

    def compute_dynamic_weights(
        self,
        query_embedding: list,
        temperature: float = None
    ) -> dict:
        """
        Compute dynamic weights based on query intent (Section 4.3).

        Uses cosine similarity between query and anchor embeddings,
        then applies softmax with temperature to get weights.
        """
        if temperature is None:
            temperature = self.SOFTMAX_TEMPERATURE

        anchors = self.get_anchor_embeddings()

        # Compute cosine similarities
        similarities = {}
        for modality, anchor_emb in anchors.items():
            similarities[modality] = cosine_similarity(query_embedding, anchor_emb)

        # Apply softmax with temperature
        weights = softmax_with_temperature(similarities, temperature)

        return {
            "weights": weights,
            "similarities": similarities
        }

    def search(
        self,
        query: str,
        modalities: Optional[list] = None,
        weights: Optional[dict] = None,
        limit: int = 50,
        video_id: Optional[str] = None,
        fusion_method: str = "rrf",  # "rrf", "weighted", or "dynamic"
        k_per_modality: int = 20
    ) -> list:
        """
        Search for video segments matching a text query.

        Args:
            query: Text search query
            modalities: List of modalities to search ["visual", "audio", "transcription"]
            weights: Weights per modality (for RRF, these weight the rank contribution)
            limit: Maximum results
            video_id: Optional filter by specific video
            fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted" (score sum)

        Returns:
            List of ranked results with fusion scores
        """
        if modalities is None:
            modalities = ["visual", "audio", "transcription"]

        if weights is None:
            weights = self.DEFAULT_WEIGHTS.copy()

        # Generate query embedding
        query_result = self.bedrock.get_text_query_embedding(query)
        query_embedding = query_result["embedding"]

        if not query_embedding:
            return []

        # Search each modality and collect ranked results
        modality_results = {}

        for modality in modalities:
            weight = weights.get(modality, 1.0)
            if weight == 0:
                continue

            filter_doc = {"modality_type": modality}
            if video_id:
                filter_doc["video_id"] = video_id

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 6,
                        "limit": limit * 2,  # Get more candidates for fusion
                        "filter": filter_doc
                    }
                },
                {
                    "$project": {
                        "video_id": 1,
                        "start_time": 1,
                        "end_time": 1,
                        "s3_uri": 1,
                        "segment_id": 1,
                        "modality_type": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            modality_results[modality] = results

        # Apply fusion
        if fusion_method == "rrf":
            return self._rrf_fusion(modality_results, weights, limit)
        else:
            return self._weighted_fusion(modality_results, weights, limit)

    def _rrf_fusion(
        self,
        modality_results: dict,
        weights: dict,
        limit: int
    ) -> list:
        """
        Reciprocal Rank Fusion (RRF).

        Formula: score(d) = Σ w_m / (k + rank_m(d))

        This is more robust than score-based fusion because:
        - Handles different score distributions across modalities
        - Emphasizes agreement between modalities
        - Standard approach used by Elasticsearch, etc.
        """
        segment_scores = {}

        for modality, results in modality_results.items():
            weight = weights.get(modality, 1.0)

            for rank, doc in enumerate(results, start=1):
                key = (doc["video_id"], doc["start_time"])

                if key not in segment_scores:
                    segment_scores[key] = {
                        "video_id": doc["video_id"],
                        "segment_id": doc.get("segment_id", 0),
                        "start_time": doc["start_time"],
                        "end_time": doc["end_time"],
                        "s3_uri": doc["s3_uri"],
                        "rrf_score": 0.0,
                        "modality_scores": {},
                        "modality_ranks": {}
                    }

                # RRF contribution: weight / (k + rank)
                rrf_contribution = weight / (self.RRF_K + rank)
                segment_scores[key]["rrf_score"] += rrf_contribution
                segment_scores[key]["modality_scores"][modality] = doc["score"]
                segment_scores[key]["modality_ranks"][modality] = rank

        # Sort by RRF score
        ranked = sorted(
            segment_scores.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        # Normalize and rename for API consistency
        for item in ranked:
            item["fusion_score"] = item.pop("rrf_score")

        return ranked[:limit]

    def _weighted_fusion(
        self,
        modality_results: dict,
        weights: dict,
        limit: int
    ) -> list:
        """
        Simple weighted score fusion.

        Formula: score(d) = Σ w_m * sim_m(d)
        """
        segment_scores = {}

        for modality, results in modality_results.items():
            for doc in results:
                key = (doc["video_id"], doc["start_time"])

                if key not in segment_scores:
                    segment_scores[key] = {
                        "video_id": doc["video_id"],
                        "segment_id": doc.get("segment_id", 0),
                        "start_time": doc["start_time"],
                        "end_time": doc["end_time"],
                        "s3_uri": doc["s3_uri"],
                        "modality_scores": {}
                    }

                segment_scores[key]["modality_scores"][modality] = doc["score"]

        # Compute weighted sum
        total_weight = sum(weights.values())
        for key, data in segment_scores.items():
            fusion_score = sum(
                (weights.get(m, 0) / total_weight) * data["modality_scores"].get(m, 0)
                for m in modality_results.keys()
            )
            data["fusion_score"] = fusion_score

        ranked = sorted(
            segment_scores.values(),
            key=lambda x: x["fusion_score"],
            reverse=True
        )

        return ranked[:limit]

    def search_dynamic(
        self,
        query: str,
        limit: int = 50,
        video_id: Optional[str] = None,
        temperature: float = None
    ) -> dict:
        """
        Search with dynamic intent-based routing (Section 4.3 of whitepaper).

        Automatically determines modality weights based on query semantics.

        Args:
            query: Text search query
            limit: Maximum results
            video_id: Optional filter by specific video
            temperature: Softmax temperature (higher = more uniform weights)

        Returns:
            Dict with 'results', 'weights', and 'similarities'
        """
        if temperature is None:
            temperature = self.SOFTMAX_TEMPERATURE

        # Generate query embedding
        query_result = self.bedrock.get_text_query_embedding(query)
        query_embedding = query_result["embedding"]

        if not query_embedding:
            return {"results": [], "weights": {}, "similarities": {}}

        # Compute dynamic weights based on query intent
        dynamic_result = self.compute_dynamic_weights(query_embedding, temperature)
        weights = dynamic_result["weights"]
        similarities = dynamic_result["similarities"]

        # Search each modality
        modalities = ["visual", "audio", "transcription"]
        modality_results = {}

        for modality in modalities:
            filter_doc = {"modality_type": modality}
            if video_id:
                filter_doc["video_id"] = video_id

            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": limit * 6,
                        "limit": limit * 2,
                        "filter": filter_doc
                    }
                },
                {
                    "$project": {
                        "video_id": 1,
                        "start_time": 1,
                        "end_time": 1,
                        "s3_uri": 1,
                        "segment_id": 1,
                        "modality_type": 1,
                        "score": {"$meta": "vectorSearchScore"}
                    }
                }
            ]

            results = list(self.collection.aggregate(pipeline))
            modality_results[modality] = results

        # Apply weighted fusion with dynamic weights
        results = self._weighted_fusion(modality_results, weights, limit)

        return {
            "results": results,
            "weights": weights,
            "similarities": similarities
        }

    def get_videos(self) -> list:
        """Get list of all indexed videos."""
        pipeline = [
            {"$group": {"_id": "$video_id", "s3_uri": {"$first": "$s3_uri"}}},
            {"$project": {"video_id": "$_id", "s3_uri": 1, "_id": 0}}
        ]
        return list(self.collection.aggregate(pipeline))

    def get_segment(self, video_id: str, segment_id: int) -> Optional[dict]:
        """Get a specific segment by video_id and segment_id."""
        return self.collection.find_one(
            {"video_id": video_id, "segment_id": segment_id},
            {"video_id": 1, "segment_id": 1, "s3_uri": 1, "start_time": 1, "end_time": 1}
        )

    def close(self):
        """Close MongoDB connection."""
        self.mongo_client.close()


def create_client(
    mongodb_uri: str,
    database_name: str = "video_search"
) -> VideoSearchClient:
    """Factory function to create a VideoSearchClient."""
    return VideoSearchClient(mongodb_uri=mongodb_uri, database_name=database_name)
