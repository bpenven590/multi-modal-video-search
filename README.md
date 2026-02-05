# Multi-Vector Video Search Pipeline

A video semantic search pipeline using AWS Bedrock Marengo 3.0 with dual vector storage backends: **MongoDB Atlas** (single-index) and **Amazon S3 Vectors** (multi-index).

## Live Demo

**Search UI:** https://nyfwaxmgni.us-east-1.awsapprunner.com

## Search UI Features

The web interface provides multiple search modes and backend options:

### Search Modes

**Fused Approaches** (multi-vector fusion):
- **RRF** - Reciprocal Rank Fusion (rank-based)
- **Weighted** - Score-based fusion with adjustable weights
- **Dynamic** - Intent-based routing with query anchors

**Single Modality**:
- **Visual** - Visual content only
- **Audio** - Audio/sound only
- **Speech** - Transcription/dialogue only

### Backend Toggle

- **MongoDB (Single Index)** - One collection with modality filter
- **S3 Vectors (Multi-Index)** - Separate index per modality

### Result Card Layout

Each search result displays ranking, confidence, and modality breakdown:

```
┌─────────────────────────────┐
│ #1           85%     [VIS]  │  ← Rank, Confidence %, Dominant Modality
│                             │
│     [Video Thumbnail]       │
│                             │
│         0:30 - 1:15         │  ← Timestamp Range
└─────────────────────────────┘
  Video Title
  vis: 0.85  aud: 0.12  tra: 0.03  ← Individual Modality Scores
  ███████░░ ███░░░░░░ █░░░░░░░░  ← Visual Score Bars
```

**Key Features:**
- **Ranking Badge** (#1, #2, #3...) - Shows result position
- **Confidence %** - Match confidence (0-100%)
- **Dominant Badge** - Which modality scored highest (VIS/AUD/TRA)
- **Modality Scores** - Detailed breakdown per embedding type
- **Score Visualization** - Visual bars showing relative strengths
- **20 Results per Page** - Focused, high-quality results

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐
│   S3 Bucket     │     │  AWS Lambda      │
│   (Videos)      │────▶│  (Processing)    │
│                 │     │                  │
│ tl-brice-media/ │     │  ┌────────────┐  │
│ WBD_project/    │     │  │  Bedrock   │  │
│ Videos/Ready/   │     │  │  Marengo   │  │
└────────┬────────┘     │  │  3.0       │  │
         │              │  └────────────┘  │
    S3 Trigger          │                  │
    (automatic)         │  Embeddings:     │
                        │  - Visual (512d) │
                        │  - Audio (512d)  │
                        │  - Transcription │
                        │    (512d)        │
                        └─────────┬────────┘
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────────────┐               ┌─────────────────────────────┐
│   MongoDB Atlas         │               │   Amazon S3 Vectors         │
│   (Single Index Mode)   │               │   (Multi-Index Mode)        │
│                         │               │                             │
│ ┌─────────────────────┐ │               │ ┌─────────────────────────┐ │
│ │  video_embeddings   │ │               │ │  visual-embeddings      │ │
│ │  (single collection)│ │               │ │  (separate index)       │ │
│ │                     │ │               │ ├─────────────────────────┤ │
│ │  modality_type:     │ │               │ │  audio-embeddings       │ │
│ │  - visual           │ │               │ │  (separate index)       │ │
│ │  - audio            │ │               │ ├─────────────────────────┤ │
│ │  - transcription    │ │               │ │  transcription-embs     │ │
│ │                     │ │               │ │  (separate index)       │ │
│ │  HNSW Vector Index  │ │               │ └─────────────────────────┘ │
│ │  + Filter Fields    │ │               │  Bucket: brice-video-       │
│ └─────────────────────┘ │               │  search-multimodal          │
└────────────┬────────────┘               └─────────────┬───────────────┘
             │                                          │
             └──────────────────┬───────────────────────┘
                                │
┌─────────────────┐     ┌───────┴──────────┐
│   CloudFront    │     │  AWS App Runner  │
│   (CDN)         │◀────│  (Search API)    │
│                 │     │                  │
│ Video streaming │     │  ┌────────────┐  │
│ + thumbnails    │     │  │  FastAPI   │  │
└─────────────────┘     │  │  + RRF     │  │
                        │  │  Fusion    │  │
                        │  └────────────┘  │
                        │                  │
                        │  UI Toggle:      │
                        │  MongoDB ↔ S3    │
                        │  Vectors         │
                        └──────────────────┘
```

## Vector Storage Backends

### MongoDB Atlas (Single Index Mode) - Default

Uses a **single collection** (`video_embeddings`) with a `modality_type` field, following the "Single index with distinguished modalities" pattern from the TwelveLabs guidance (Section 3.2.1).

**Benefits:**
- Pre-filter by `modality_type` to search specific modalities
- Search all modalities in one query
- Simpler index management
- Flexible fusion strategies
- Free tier available (M0)

**Limitation:** MongoDB Atlas free tier allows only 3 vector indexes.

### Amazon S3 Vectors (Multi-Index Mode)

Uses **separate indexes** per modality in a dedicated S3 vector bucket (`brice-video-search-multimodal`), providing true multi-index search.

**Benefits:**
- Dedicated index per modality (no filter overhead)
- Potentially faster for modality-specific queries
- Native AWS integration
- No index count limitations

**Indexes:**
- `visual-embeddings` - 512-dimensional visual embeddings
- `audio-embeddings` - 512-dimensional audio embeddings
- `transcription-embeddings` - 512-dimensional transcription embeddings

### Reciprocal Rank Fusion (RRF)

Instead of simple weighted sum, this implementation uses **RRF** for more robust multi-modal fusion:

```
score(d) = Σ w_m / (k + rank_m(d))
```

Where:
- `k = 60` (standard RRF constant)
- `w_m` = modality weight
- `rank_m(d)` = rank of document d in modality m's results

Default weights: `visual=0.8, audio=0.1, transcription=0.05`

RRF is rank-based rather than score-based, making it more robust to score distribution differences across modalities.

---

## Project Structure

```
multi-modal-video-search/
├── app.py                        # FastAPI web application (search API)
├── src/
│   ├── lambda_function.py        # Lambda handler for video processing
│   ├── bedrock_client.py         # Bedrock Marengo client
│   ├── mongodb_client.py         # MongoDB embedding storage
│   ├── s3_vectors_client.py      # S3 Vectors embedding storage & search
│   ├── search_client.py          # Video search client with RRF fusion
│   └── query_fusion.py           # Legacy query fusion script
├── static/
│   └── index.html                # Search UI frontend (MongoDB/S3 Vectors toggle)
├── scripts/
│   ├── deploy.sh                 # AWS CLI deployment script
│   ├── mongodb_setup.md          # MongoDB Atlas setup guide
│   └── migrate_to_s3_vectors.py  # Migration script: MongoDB → S3 Vectors
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

---

## Prerequisites

- **AWS Account** with access to:
  - AWS Lambda
  - AWS Bedrock (Marengo model enabled in us-east-1)
  - AWS App Runner
  - S3 (read access to video bucket)
  - CloudFront (optional, for video CDN)
- **MongoDB Atlas** account (free tier M0 works)
- **Python 3.11+**
- **AWS CLI** configured with appropriate credentials

---

## Quick Start

### 1. Clone and Setup

```bash
cd multi-modal-video-search

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

1. Create a cluster (free tier M0 works)
2. Create database user and get connection string
3. Create the `video_embeddings` collection with vector index
4. Whitelist IPs (or use 0.0.0.0/0 for testing)
5. Update `MONGODB_URI` in your `.env` file

### 2b. Setup S3 Vectors (Optional - Multi-Index Mode)

To enable the S3 Vectors backend for multi-index search:

```bash
# Create vector bucket
aws s3vectors create-vector-bucket \
  --vector-bucket-name brice-video-search-multimodal \
  --region us-east-1

# Create indexes for each modality (512 dimensions, cosine similarity)
for index in visual-embeddings audio-embeddings transcription-embeddings; do
  aws s3vectors create-index \
    --vector-bucket-name brice-video-search-multimodal \
    --index-name $index \
    --data-type float32 \
    --dimension 512 \
    --distance-metric cosine \
    --metadata-configuration '{
      "nonFilterableMetadataKeys": ["video_id", "segment_id", "s3_uri", "start_time", "end_time"]
    }' \
    --region us-east-1
done

# Migrate existing embeddings from MongoDB to S3 Vectors
MONGODB_URI='your_mongodb_uri' python scripts/migrate_to_s3_vectors.py
```

### 3. Deploy Lambda Function

```bash
# Set MongoDB URI
export MONGODB_URI="your_mongodb_connection_string_here"

# Deploy
./scripts/deploy.sh
```

### 4. Video Storage Structure

Videos are stored in S3 with the following structure:
- **Original files**: `s3://tl-brice-media/WBD_project/Videos/` (1080p, full quality)
- **Proxy files**: `s3://tl-brice-media/WBD_project/Videos/proxy/` (480p, for fast thumbnails)

Proxy files are generated using AWS MediaConvert for faster thumbnail loading.

### 5. Setup S3 Trigger (Automatic Processing)

Configure S3 to automatically trigger Lambda when videos are uploaded:

```bash
# Add S3 trigger for the Ready folder
aws s3api put-bucket-notification-configuration \
  --bucket tl-brice-media \
  --notification-configuration '{
    "LambdaFunctionConfigurations": [{
      "Id": "marengo-embedding-trigger",
      "LambdaFunctionArn": "arn:aws:lambda:us-east-1:ACCOUNT_ID:function:video-embedding-pipeline",
      "Events": ["s3:ObjectCreated:*"],
      "Filter": {
        "Key": {
          "FilterRules": [
            {"Name": "prefix", "Value": "WBD_project/Videos/Ready/"},
            {"Name": "suffix", "Value": ".mp4"}
          ]
        }
      }
    }]
  }'
```

### 5. Deploy Search API (App Runner)

The search API runs on AWS App Runner with auto-deploy from GitHub:

1. Create App Runner service connected to your GitHub repo
2. Set environment variables:
   - `MONGODB_URI` - MongoDB connection string
   - `CLOUDFRONT_DOMAIN` - CloudFront distribution domain
   - `PYTHONPATH` - `/app/deps` (for dependencies)
3. Build command: `pip3 install -r requirements.txt -t ./deps`
4. Start command: `python3 app.py`

### 6. Process a Video

**Manual invocation:**
```bash
aws lambda invoke \
  --function-name video-embedding-pipeline \
  --region us-east-1 \
  --payload '{"s3_key": "WBD_project/Videos/Ready/sample.mp4", "bucket": "tl-brice-media"}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

**Or simply upload to S3 Ready folder** (with S3 trigger configured):
```bash
aws s3 cp video.mp4 s3://tl-brice-media/WBD_project/Videos/Ready/
```

### 7. Search Videos

**Via Web UI:** https://nyfwaxmgni.us-east-1.awsapprunner.com

**Via API:**
```bash
curl "https://nyfwaxmgni.us-east-1.awsapprunner.com/api/search?q=someone+walking&limit=10"
```

---

## Lambda Event Format

The Lambda function accepts events in two formats:

**S3 Trigger (automatic):**
```json
{
  "Records": [{
    "s3": {
      "bucket": {"name": "tl-brice-media"},
      "object": {"key": "WBD_project/Videos/Ready/file.mp4"}
    }
  }]
}
```

**Manual invocation:**
```json
{
  "s3_key": "WBD_project/Videos/Ready/file.mp4",
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
| `WEIGHT_TRANSCRIPTION` | 0.05 | Weight for transcription modality |

### Recommended Weight Configurations

| Use Case | Visual | Audio | Transcription |
|----------|--------|-------|---------------|
| Visual-heavy (action, scenes) | 0.8 | 0.1 | 0.05 |
| Dialogue-focused | 0.3 | 0.1 | 0.6 |
| Audio events (music, sounds) | 0.3 | 0.5 | 0.2 |
| Balanced search | 0.4 | 0.3 | 0.3 |

---

## API Reference

### VideoSearchClient (search_client.py)

```python
from src.search_client import VideoSearchClient

client = VideoSearchClient(
    mongodb_uri="mongodb+srv://...",
    database_name="video_search"
)

# RRF fusion search with custom weights
results = client.search(
    query="a person running",
    limit=10,
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.05}
)

# Returns list of results with fusion scores and per-modality scores
for r in results:
    print(f"{r['video_id']} seg {r['segment_id']}: {r['fusion_score']}")
    print(f"  Modality scores: {r['modality_scores']}")
```

### BedrockMarengoClient (bedrock_client.py)

```python
from src.bedrock_client import BedrockMarengoClient

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

### MongoDBEmbeddingClient (mongodb_client.py)

```python
from src.mongodb_client import MongoDBEmbeddingClient

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
    modality_filter="visual"
)
```

### S3VectorsClient (s3_vectors_client.py)

```python
from src.s3_vectors_client import S3VectorsClient

client = S3VectorsClient(
    bucket_name="brice-video-search-multimodal",
    region="us-east-1"
)

# Store embeddings (writes to all 3 modality indexes)
result = client.store_all_segments(video_id="abc123", segments=segments)

# Search with fusion across modalities
results = client.search_with_fusion(
    query_embedding=embedding,
    modalities=["visual", "audio", "transcription"],
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.05},
    limit=10,
    fusion_method="rrf"  # or "weighted"
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

1. Verify MongoDB Atlas IP whitelist includes Lambda/App Runner IPs
2. Check connection string format
3. For testing, use 0.0.0.0/0 in Atlas Network Access

### App Runner Build Fails

- Use `pip3` instead of `pip` in build command
- Install to `./deps` directory and set `PYTHONPATH=/app/deps`

---

## References

- [TwelveLabs Multi-Vector Guidance](./A%20Guidance%20on%20Multi-Vector%20Video%20Search%20with%20TwelveLabs%20Marengo.pdf) - Section 3.2.1 (Single index with distinguished modalities)
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Amazon S3 Vectors Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

---

## License

Internal use only. All rights reserved.
