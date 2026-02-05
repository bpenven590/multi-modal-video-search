#!/usr/bin/env python3
"""
Migrate embeddings from MongoDB to S3 Vectors.

Reads from the video_embeddings collection in MongoDB and writes to
modality-specific indexes in S3 Vectors.
"""

import os
import sys
from pymongo import MongoClient
import boto3

# Configuration
MONGODB_URI = os.environ.get("MONGODB_URI")
S3_VECTORS_BUCKET = "brice-video-search-multimodal"
AWS_REGION = "us-east-1"
BATCH_SIZE = 100  # S3 Vectors supports batch writes

# Index mapping
INDEX_NAMES = {
    "visual": "visual-embeddings",
    "audio": "audio-embeddings",
    "transcription": "transcription-embeddings"
}


def migrate():
    if not MONGODB_URI:
        print("Error: MONGODB_URI environment variable required")
        sys.exit(1)

    # Connect to MongoDB
    print("Connecting to MongoDB...")
    mongo_client = MongoClient(MONGODB_URI)
    db = mongo_client["video_search"]
    collection = db["video_embeddings"]

    # Connect to S3 Vectors using SSO profile (for local dev)
    print("Connecting to S3 Vectors...")
    aws_profile = os.environ.get("AWS_PROFILE", "TlFullDevelopmentAccess-026090552520")
    if aws_profile:
        session = boto3.Session(profile_name=aws_profile)
        s3v_client = session.client("s3vectors", region_name=AWS_REGION)
    else:
        s3v_client = boto3.client("s3vectors", region_name=AWS_REGION)

    # Get total count per modality
    print("\nCounting documents in MongoDB...")
    for modality in INDEX_NAMES.keys():
        count = collection.count_documents({"modality_type": modality})
        print(f"  {modality}: {count} documents")

    # Migrate each modality
    for modality, index_name in INDEX_NAMES.items():
        print(f"\n{'='*50}")
        print(f"Migrating {modality} embeddings to {index_name}...")
        print("="*50)

        # Query MongoDB for this modality
        cursor = collection.find(
            {"modality_type": modality},
            {
                "video_id": 1,
                "segment_id": 1,
                "s3_uri": 1,
                "start_time": 1,
                "end_time": 1,
                "embedding": 1
            }
        )

        batch = []
        total_migrated = 0
        errors = 0

        for doc in cursor:
            vector_key = f"{doc['video_id']}_{doc['segment_id']}"

            vector_data = {
                "key": vector_key,
                "data": {"float32": doc["embedding"]},
                "metadata": {
                    "video_id": doc["video_id"],
                    "segment_id": str(doc["segment_id"]),
                    "s3_uri": doc["s3_uri"],
                    "start_time": str(doc["start_time"]),
                    "end_time": str(doc["end_time"])
                }
            }

            batch.append(vector_data)

            # Write batch when full
            if len(batch) >= BATCH_SIZE:
                try:
                    s3v_client.put_vectors(
                        vectorBucketName=S3_VECTORS_BUCKET,
                        indexName=index_name,
                        vectors=batch
                    )
                    total_migrated += len(batch)
                    print(f"  Migrated {total_migrated} vectors...", end="\r")
                except Exception as e:
                    print(f"\n  Error writing batch: {e}")
                    errors += len(batch)
                batch = []

        # Write remaining batch
        if batch:
            try:
                s3v_client.put_vectors(
                    vectorBucketName=S3_VECTORS_BUCKET,
                    indexName=index_name,
                    vectors=batch
                )
                total_migrated += len(batch)
            except Exception as e:
                print(f"\n  Error writing final batch: {e}")
                errors += len(batch)

        print(f"\n  Completed: {total_migrated} vectors migrated, {errors} errors")

    # Summary
    print("\n" + "="*50)
    print("Migration complete!")
    print("="*50)

    # Verify by listing indexes
    print("\nVerifying S3 Vectors indexes...")
    response = s3v_client.list_indexes(vectorBucketName=S3_VECTORS_BUCKET)
    for idx in response.get("indexes", []):
        print(f"  {idx['indexName']}: created {idx['creationTime']}")

    mongo_client.close()


if __name__ == "__main__":
    migrate()
