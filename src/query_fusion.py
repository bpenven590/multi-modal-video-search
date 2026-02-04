"""
Query Fusion Testing Script for Multi-Vector Video Search

Implements score-based fusion (Equation 3 from TwelveLabs Multi-Vector Guidance):

    score(s) = Σ w_m · sim(Q_m, E_m(s))

Where:
    - s represents the video segment
    - m represents a vector modality in M = [visual, audio, transcription]
    - w_m represents the weight for the vector modality m
    - Q_m is the query embedding for modality m
    - E_m(s) is the m modality embedding of video segment s
    - sim(A, B) represents the cosine similarity between 2 vectors A and B

Default weights: visual=0.8, audio=0.1, transcription=0.1
Configurable via environment variables:
    - WEIGHT_VISUAL (default: 0.8)
    - WEIGHT_AUDIO (default: 0.1)
    - WEIGHT_TRANSCRIPTION (default: 0.1)

Single Collection Approach:
    All embeddings stored in one collection with modality_type field.
    Enables pre-filtering by modality or searching all at once.
"""

import os
import json
import argparse
from collections import defaultdict
from typing import Optional, List
from dotenv import load_dotenv

from bedrock_client import BedrockMarengoClient
from mongodb_client import MongoDBEmbeddingClient


class QueryFusionSearch:
    """
    Multi-vector search with score-based fusion.

    Implements Section 3.2.3 (Ranking) from TwelveLabs Multi-Vector Guidance.
    Uses single collection with modality_type filtering.
    """

    MODALITY_TYPES = ["visual", "audio", "transcription"]

    def __init__(
        self,
        mongodb_client: MongoDBEmbeddingClient,
        bedrock_client: BedrockMarengoClient,
        weight_visual: float = 0.8,
        weight_audio: float = 0.1,
        weight_transcription: float = 0.1
    ):
        """
        Initialize the query fusion search.

        Args:
            mongodb_client: MongoDB client for vector search
            bedrock_client: Bedrock client for query embedding
            weight_visual: Weight for visual modality (default: 0.8)
            weight_audio: Weight for audio modality (default: 0.1)
            weight_transcription: Weight for transcription modality (default: 0.1)
        """
        self.mongodb = mongodb_client
        self.bedrock = bedrock_client

        # Modality weights (should sum to 1.0 for normalized scoring)
        self.weights = {
            "visual": weight_visual,
            "audio": weight_audio,
            "transcription": weight_transcription
        }

        # Validate weights
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            print(f"Warning: Weights sum to {total_weight}, not 1.0. "
                  "Scores may not be normalized.")

    def search(
        self,
        query_text: str,
        limit: int = 10,
        per_modality_limit: int = 50,
        modalities: Optional[List[str]] = None,
        video_id_filter: Optional[str] = None
    ) -> list:
        """
        Perform multi-vector search with score-based fusion.

        Args:
            query_text: Natural language search query
            limit: Maximum number of final results
            per_modality_limit: Results to retrieve per modality before fusion
            modalities: Modalities to search (default: all three)
            video_id_filter: Optional filter to search within specific video

        Returns:
            List of fused search results, sorted by combined score
        """
        if modalities is None:
            modalities = self.MODALITY_TYPES

        # Step 1: Generate query embedding
        print(f"Generating query embedding for: '{query_text}'")
        query_result = self.bedrock.get_text_query_embedding(query_text)
        query_embedding = query_result["embedding"]

        # Step 2: Search each modality using pre-filtering
        print(f"Searching modalities: {modalities}")
        modality_results = self.mongodb.multi_modality_search(
            query_embedding=query_embedding,
            limit_per_modality=per_modality_limit,
            modalities=modalities,
            video_id_filter=video_id_filter
        )

        # Step 3: Apply score-based fusion (Equation 3)
        fused_results = self._apply_score_fusion(modality_results, modalities)

        # Step 4: Sort by fused score and return top results
        sorted_results = sorted(
            fused_results.values(),
            key=lambda x: x["fused_score"],
            reverse=True
        )

        return sorted_results[:limit]

    def search_single_modality(
        self,
        query_text: str,
        modality: str,
        limit: int = 10,
        video_id_filter: Optional[str] = None
    ) -> list:
        """
        Search a single modality without fusion (direct pre-filtered search).

        Args:
            query_text: Natural language search query
            modality: Modality to search ("visual", "audio", "transcription")
            limit: Maximum number of results
            video_id_filter: Optional filter to search within specific video

        Returns:
            List of search results from single modality
        """
        if modality not in self.MODALITY_TYPES:
            raise ValueError(f"Invalid modality: {modality}. Must be one of {self.MODALITY_TYPES}")

        # Generate query embedding
        print(f"Generating query embedding for: '{query_text}'")
        query_result = self.bedrock.get_text_query_embedding(query_text)
        query_embedding = query_result["embedding"]

        # Search single modality with pre-filter
        print(f"Searching modality: {modality}")
        results = self.mongodb.vector_search(
            query_embedding=query_embedding,
            limit=limit,
            modality_filter=modality,
            video_id_filter=video_id_filter
        )

        return results

    def _apply_score_fusion(
        self,
        modality_results: dict,
        modalities: List[str]
    ) -> dict:
        """
        Apply score-based fusion across modality search results.

        Implements Equation 3: score(s) = Σ w_m · sim(Q_m, E_m(s))

        Args:
            modality_results: Dictionary of results per modality
            modalities: List of modalities that were searched

        Returns:
            Dictionary of segment keys to fused result data
        """
        # Group results by unique segment (video_id + segment_id)
        segment_scores = defaultdict(lambda: {
            "video_id": None,
            "segment_id": None,
            "s3_uri": None,
            "start_time": None,
            "end_time": None,
            "modality_scores": {},
            "fused_score": 0.0
        })

        for modality in modalities:
            results = modality_results.get(modality, [])
            weight = self.weights.get(modality, 0.0)

            for result in results:
                # Create unique segment key
                segment_key = f"{result['video_id']}_{result['segment_id']}"

                # Update segment data if not set
                if segment_scores[segment_key]["video_id"] is None:
                    segment_scores[segment_key]["video_id"] = result["video_id"]
                    segment_scores[segment_key]["segment_id"] = result["segment_id"]
                    segment_scores[segment_key]["s3_uri"] = result["s3_uri"]
                    segment_scores[segment_key]["start_time"] = result["start_time"]
                    segment_scores[segment_key]["end_time"] = result["end_time"]

                # Store modality-specific score
                modality_score = result.get("score", 0.0)
                segment_scores[segment_key]["modality_scores"][modality] = modality_score

                # Apply weighted score (Equation 3)
                segment_scores[segment_key]["fused_score"] += weight * modality_score

        return segment_scores

    def search_with_details(
        self,
        query_text: str,
        limit: int = 10,
        per_modality_limit: int = 50,
        modalities: Optional[List[str]] = None,
        video_id_filter: Optional[str] = None
    ) -> dict:
        """
        Perform search and return detailed results including per-modality scores.

        Args:
            query_text: Natural language search query
            limit: Maximum number of final results
            per_modality_limit: Results to retrieve per modality
            modalities: Modalities to search
            video_id_filter: Optional filter to search within specific video

        Returns:
            Dictionary with results and metadata
        """
        results = self.search(
            query_text=query_text,
            limit=limit,
            per_modality_limit=per_modality_limit,
            modalities=modalities,
            video_id_filter=video_id_filter
        )

        return {
            "query": query_text,
            "weights": self.weights,
            "modalities_searched": modalities or self.MODALITY_TYPES,
            "video_id_filter": video_id_filter,
            "result_count": len(results),
            "results": results
        }


def get_weights_from_env() -> dict:
    """
    Load modality weights from environment variables.

    Environment variables:
        - WEIGHT_VISUAL (default: 0.8)
        - WEIGHT_AUDIO (default: 0.1)
        - WEIGHT_TRANSCRIPTION (default: 0.1)

    Returns:
        Dictionary of modality weights
    """
    return {
        "visual": float(os.environ.get("WEIGHT_VISUAL", "0.8")),
        "audio": float(os.environ.get("WEIGHT_AUDIO", "0.1")),
        "transcription": float(os.environ.get("WEIGHT_TRANSCRIPTION", "0.1"))
    }


def main():
    """CLI entry point for query fusion testing."""
    parser = argparse.ArgumentParser(
        description="Multi-vector video search with score-based fusion"
    )
    parser.add_argument(
        "query",
        type=str,
        help="Search query text"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results (default: 10)"
    )
    parser.add_argument(
        "--visual-weight",
        type=float,
        default=None,
        help="Weight for visual modality (overrides env var)"
    )
    parser.add_argument(
        "--audio-weight",
        type=float,
        default=None,
        help="Weight for audio modality (overrides env var)"
    )
    parser.add_argument(
        "--transcription-weight",
        type=float,
        default=None,
        help="Weight for transcription modality (overrides env var)"
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="visual,audio,transcription",
        help="Comma-separated list of modalities to search"
    )
    parser.add_argument(
        "--single-modality",
        type=str,
        choices=["visual", "audio", "transcription"],
        default=None,
        help="Search only a single modality (no fusion)"
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Filter results to a specific video ID"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Get weights (CLI args override env vars)
    weights = get_weights_from_env()
    if args.visual_weight is not None:
        weights["visual"] = args.visual_weight
    if args.audio_weight is not None:
        weights["audio"] = args.audio_weight
    if args.transcription_weight is not None:
        weights["transcription"] = args.transcription_weight

    # Parse modalities
    modalities = [m.strip() for m in args.modalities.split(",")]

    # Initialize clients
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        print("Error: MONGODB_URI environment variable not set")
        return 1

    mongodb_client = MongoDBEmbeddingClient(
        connection_string=mongodb_uri,
        database_name=os.environ.get("MONGODB_DATABASE", "video_search")
    )

    bedrock_client = BedrockMarengoClient(region="us-east-1")

    # Create search instance
    search = QueryFusionSearch(
        mongodb_client=mongodb_client,
        bedrock_client=bedrock_client,
        weight_visual=weights["visual"],
        weight_audio=weights["audio"],
        weight_transcription=weights["transcription"]
    )

    # Execute search
    if args.single_modality:
        # Single modality search (no fusion)
        print(f"\nSearching single modality: {args.single_modality}")
        print(f"Query: {args.query}\n")

        results = search.search_single_modality(
            query_text=args.query,
            modality=args.single_modality,
            limit=args.limit,
            video_id_filter=args.video_id
        )

        if args.json:
            print(json.dumps({
                "query": args.query,
                "modality": args.single_modality,
                "result_count": len(results),
                "results": results
            }, indent=2, default=str))
        else:
            print(f"Found {len(results)} results:\n")
            print("-" * 80)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Score: {result.get('score', 0):.4f}")
                print(f"   Video ID: {result['video_id']}")
                print(f"   Segment: {result['segment_id']} "
                      f"({result['start_time']:.2f}s - {result['end_time']:.2f}s)")
                print(f"   S3 URI: {result['s3_uri']}")
                print(f"   Modality: {result['modality_type']}")
    else:
        # Multi-modality fusion search
        print(f"\nSearching with weights: {weights}")
        print(f"Modalities: {modalities}")
        print(f"Query: {args.query}\n")

        results = search.search_with_details(
            query_text=args.query,
            limit=args.limit,
            modalities=modalities,
            video_id_filter=args.video_id
        )

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"Found {results['result_count']} results:\n")
            print("-" * 80)

            for i, result in enumerate(results["results"], 1):
                print(f"\n{i}. Fused Score: {result['fused_score']:.4f}")
                print(f"   Video ID: {result['video_id']}")
                print(f"   Segment: {result['segment_id']} "
                      f"({result['start_time']:.2f}s - {result['end_time']:.2f}s)")
                print(f"   S3 URI: {result['s3_uri']}")
                print("   Modality Scores:")
                for modality, score in result["modality_scores"].items():
                    weight = weights.get(modality, 0)
                    weighted = weight * score
                    print(f"      {modality}: {score:.4f} (weight: {weight}, "
                          f"contribution: {weighted:.4f})")

    # Cleanup
    mongodb_client.close()

    return 0


if __name__ == "__main__":
    exit(main())
