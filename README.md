# Multi-Vector Video Search Pipeline

A video semantic search pipeline using AWS Bedrock Marengo 3.0 and MongoDB Atlas with multi-vector retrieval and score-based fusion.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   S3 Bucket     │     │  AWS Lambda      │     │   MongoDB Atlas         │
│   (Videos)      │────▶│  (Processing)    │────▶│   (us-east-1)           │
│                 │     │                  │     │                         │
│ tl-brice-media/ │     │  ┌────────────┐  │     │ ┌─────────────────────┐ │
│ WBD_project/    │     │  │  Bedrock   │  │     │ │  video_embeddings   │ │
│ Videos/         │     │  │  Marengo   │  │     │ │  (single collection)│ │
└─────────────────┘     │  │  3.0       │  │     │ │                     │ │
                        │  └────────────┘  │     │ │  modality_type:     │ │
                        │                  │     │ │  - visual           │ │
                        │  Embeddings:     │     │ │  - audio            │ │
                        │  - Visual (512d) │     │ │  - transcription    │ │
                        │  - Audio (512d)  │     │ │                     │ │
                        │  - Transcription │     │ │  HNSW Vector Index  │ │
                        │    (512d)        │     │ │  + Filter Fields    │ │
                        └──────────────────┘     │ └─────────────────────┘ │
                                                 └─────────────────────────┘
```

### Single Collection with Modality Filtering

This implementation uses a **single collection** (`video_embeddings`) with a `modality_type` field, following the "Single index with distinguished modalities" pattern from the TwelveLabs guidance (Section 3.2.1).

**Benefits:**
- Pre-filter by `modality_type` to search specific modalities
- Search all modalities in one query
- Simpler index management
- Flexible fusion strategies (weighted, anchor-based, or direct modality selection)

### Score-Based Fusion (Equation 3)

```
score(s) = w_visual × sim(Q, E_visual) + w_audio × sim(Q, E_audio) + w_transcription × sim(Q, E_transcription)
```

Default weights: `visual=0.8, audio=0.1, transcription=0.1`

---

## Project Structure

```
s3-marengo-mongodb-pipeline/
├── src/
│   ├── lambda_function.py    # Lambda handler for video processing
│   ├── bedrock_client.py     # Bedrock Marengo client
│   ├── mongodb_client.py     # MongoDB embedding storage (single collection)
│   └── query_fusion.py       # Query fusion testing script
├── scripts/
│   ├── deploy.sh             # AWS CLI deployment script
│   └── mongodb_setup.md      # MongoDB Atlas setup guide
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
└── README.md                 # This file
```

---

## Prerequisites

- **AWS Account** with access to:
  - AWS Lambda
  - AWS Bedrock (Marengo model enabled)
  - S3 (read access to video bucket)
- **MongoDB Atlas** account with M10+ cluster
- **Python 3.11+**
- **AWS CLI** configured with appropriate credentials

---

## Quick Start

### 1. Clone and Setup

```bash
cd s3-marengo-mongodb-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your MongoDB URI and other settings
```

### 2. Setup MongoDB Atlas

Follow the detailed guide in [scripts/mongodb_setup.md](scripts/mongodb_setup.md):

1. Create an M10+ cluster in **us-east-1**
2. Create database user and get connection string
3. Create the `video_embeddings` collection with vector index
4. Update `MONGODB_URI` in your `.env` file

### 3. Deploy Lambda Function

```bash
# Set MongoDB URI
export MONGODB_URI="your_mongodb_connection_string_here"

# Deploy
./scripts/deploy.sh
```

### 4. Process a Video

```bash
# Invoke Lambda to process a video
aws lambda invoke \
  --function-name video-embedding-pipeline \
  --region us-east-1 \
  --payload '{"s3_key": "WBD_project/Videos/sample.mp4", "bucket": "tl-brice-media"}' \
  --cli-binary-format raw-in-base64-out \
  response.json

cat response.json
```

### 5. Run Search Queries

```bash
# Fusion search with default weights (0.8/0.1/0.1)
python src/query_fusion.py "a person walking on the beach"

# Single modality search (no fusion, pre-filtered)
python src/query_fusion.py "someone saying hello" --single-modality transcription

# Custom weights for dialogue-heavy search
python src/query_fusion.py "talking about quarterly revenue" \
  --visual-weight 0.2 \
  --audio-weight 0.2 \
  --transcription-weight 0.6

# Output as JSON
python src/query_fusion.py "explosion scene" --json --limit 5
```

---

## Lambda Event Format

The Lambda function accepts events in this format:

```json
{
  "s3_key": "WBD_project/Videos/file.mp4",
  "bucket": "tl-brice-media",
  "video_id": "optional-custom-id",
  "embedding_types": ["visual", "audio", "transcription"]
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `s3_key` | Yes | S3 object key for the video |
| `bucket` | Yes | S3 bucket name |
| `video_id` | No | Custom video identifier (auto-generated if not provided) |
| `embedding_types` | No | List of embedding types (defaults to all three) |

---

## MongoDB Schema

### Single Collection: `video_embeddings`

All embeddings stored in one collection with `modality_type` field for filtering.

### Document Schema

```json
{
  "_id": "ObjectId",
  "video_id": "string - unique video identifier",
  "segment_id": "int - segment index within video",
  "modality_type": "string - 'visual' | 'audio' | 'transcription'",
  "s3_uri": "string - s3://bucket/key",
  "embedding": "[float] - 512-dimensional vector",
  "start_time": "float - segment start (seconds)",
  "end_time": "float - segment end (seconds)",
  "created_at": "datetime - document creation time"
}
```

### Vector Index Definition

```json
{
  "fields": [
    { "type": "vector", "path": "embedding", "numDimensions": 512, "similarity": "cosine" },
    { "type": "filter", "path": "modality_type" },
    { "type": "filter", "path": "video_id" }
  ]
}
```

---

## Query Fusion Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHT_VISUAL` | 0.8 | Weight for visual modality |
| `WEIGHT_AUDIO` | 0.1 | Weight for audio modality |
| `WEIGHT_TRANSCRIPTION` | 0.1 | Weight for transcription modality |

### Recommended Weight Configurations

| Use Case | Visual | Audio | Transcription |
|----------|--------|-------|---------------|
| Visual-heavy (action, scenes) | 0.8 | 0.1 | 0.1 |
| Dialogue-focused | 0.3 | 0.1 | 0.6 |
| Audio events (music, sounds) | 0.3 | 0.5 | 0.2 |
| Balanced search | 0.4 | 0.3 | 0.3 |

### Search Modes

**Fusion Search (default):** Searches all modalities, applies weighted fusion
```bash
python src/query_fusion.py "a car driving fast"
```

**Single Modality:** Pre-filters to specific modality, no fusion
```bash
python src/query_fusion.py "crowd cheering" --single-modality audio
```

---

## API Reference

### BedrockMarengoClient

```python
from bedrock_client import BedrockMarengoClient

client = BedrockMarengoClient(region="us-east-1")

# Generate embeddings from video
result = client.get_video_embeddings(
    bucket="tl-brice-media",
    s3_key="WBD_project/Videos/file.mp4",
    embedding_types=["visual", "audio", "transcription"]
)

# Generate query embedding
query_result = client.get_text_query_embedding("a car driving fast")
```

### MongoDBEmbeddingClient

```python
from mongodb_client import MongoDBEmbeddingClient

client = MongoDBEmbeddingClient(
    connection_string="mongodb+srv://...",
    database_name="video_search"
)

# Store embeddings
result = client.store_all_segments(video_id="abc123", segments=segments)

# Vector search with modality filter
results = client.vector_search(
    query_embedding=embedding,
    limit=10,
    modality_filter="visual"  # Pre-filter to visual only
)

# Search all modalities (for fusion)
results = client.multi_modality_search(
    query_embedding=embedding,
    limit_per_modality=50,
    modalities=["visual", "audio", "transcription"]
)
```

### QueryFusionSearch

```python
from query_fusion import QueryFusionSearch

search = QueryFusionSearch(
    mongodb_client=mongodb_client,
    bedrock_client=bedrock_client,
    weight_visual=0.8,
    weight_audio=0.1,
    weight_transcription=0.1
)

# Fusion search
results = search.search(query_text="a person running", limit=10)

# Single modality search (no fusion)
results = search.search_single_modality(
    query_text="someone talking",
    modality="transcription",
    limit=10
)
```

---

## Cost Estimation

Based on Marengo 3.0 pricing:

| Component | Price |
|-----------|-------|
| Video embedding | $0.0007/second |
| Text query embedding | Included |

**Example**: 1 hour of video = 3,600 seconds × $0.0007 = **$2.52**

---

## Troubleshooting

### Lambda Timeout

- Default timeout is 15 minutes (900 seconds)
- For very long videos (>2 hours), consider splitting into segments
- Increase memory to 2048MB or higher for faster processing

### Vector Search Returns No Results

1. Verify index is in **Active** state in Atlas UI
2. Check embedding dimensions match (512)
3. Ensure collection has documents
4. Verify filter field values match exactly

### Connection Errors

1. Verify MongoDB Atlas IP whitelist includes Lambda IPs
2. Check connection string format
3. For production, use VPC peering

---

## References

- [TwelveLabs Multi-Vector Guidance](./A%20Guidance%20on%20Multi-Vector%20Video%20Search%20with%20TwelveLabs%20Marengo.pdf) - Section 3.2.1 (Single index with distinguished modalities)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

---

## License

Internal use only. All rights reserved.
