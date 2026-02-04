"""Quick test script to verify Bedrock Marengo async video embedding."""

import sys
sys.path.insert(0, "src")

from bedrock_client import BedrockMarengoClient

# Configuration
BUCKET = "tl-brice-media"
S3_KEY = "WBD_project/Videos/TEST_WBD.mp4"
REGION = "us-east-1"

def test_bedrock_marengo_async():
    print(f"Testing Bedrock Marengo 3.0 (Async) with:")
    print(f"  s3://{BUCKET}/{S3_KEY}")
    print("-" * 60)

    client = BedrockMarengoClient(
        region=REGION,
        output_bucket=BUCKET,
        output_prefix="embeddings/"  # Same as working Lambda
    )

    try:
        result = client.get_video_embeddings(
            bucket=BUCKET,
            s3_key=S3_KEY,
            embedding_types=["visual", "audio", "transcription"]
        )

        print("\nSUCCESS!")
        print("-" * 60)
        print(f"Total segments: {result['metadata']['total_segments']}")
        print(f"Model: {result['metadata']['model_id']}")
        print(f"Embedding dimension: {result['metadata']['embedding_dimension']}")

        if result["segments"]:
            print(f"\nFirst 3 segments:")
            for seg in result["segments"][:3]:
                print(f"\n  Segment {seg['segment_id']}:")
                print(f"    Time: {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
                for emb_type, emb in seg["embeddings"].items():
                    print(f"    {emb_type}: {len(emb)}d, first 3: {emb[:3]}")

        return True

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bedrock_marengo_async()
