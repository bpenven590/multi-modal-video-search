"""
Multi-Modal Video Search Client

Performs fusion search across visual, audio, and transcription
embeddings stored in MongoDB Atlas.

Supports two fusion methods:
1. Reciprocal Rank Fusion (RRF) - default, more robust
2. Weighted Score Fusion - simple weighted sum

RRF formula: score(d) = Σ w_m / (k + rank_m(d))
"""

from typing import Optional
from pymongo import MongoClient

from bedrock_client import BedrockMarengoClient


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

    def search(
        self,
        query: str,
        modalities: Optional[list] = None,
        weights: Optional[dict] = None,
        limit: int = 50,
        video_id: Optional[str] = None,
        fusion_method: str = "rrf"  # "rrf" or "weighted"
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

    def get_videos(self) -> list:
        """Get list of all indexed videos."""
        pipeline = [
            {"$group": {"_id": "$video_id", "s3_uri": {"$first": "$s3_uri"}}},
            {"$project": {"video_id": "$_id", "s3_uri": 1, "_id": 0}}
        ]
        return list(self.collection.aggregate(pipeline))

    def close(self):
        """Close MongoDB connection."""
        self.mongo_client.close()


def create_client(
    mongodb_uri: str,
    database_name: str = "video_search"
) -> VideoSearchClient:
    """Factory function to create a VideoSearchClient."""
    return VideoSearchClient(mongodb_uri=mongodb_uri, database_name=database_name)
