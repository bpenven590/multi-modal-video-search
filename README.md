# Multi-Vector Video Search Pipeline

A production-grade video semantic search pipeline implementing the [TwelveLabs Multi-Vector Video Search Guidance](./A%20Guidance%20on%20Multi-Vector%20Video%20Search%20with%20TwelveLabs%20Marengo.pdf). Built with AWS Bedrock Marengo 3.0, featuring dual vector storage backends: **MongoDB Atlas** (single-index) and **Amazon S3 Vectors** (multi-index).

## Live Demo

**Search UI:** https://nyfwaxmgni.us-east-1.awsapprunner.com

---

## 🎯 Multi-Vector Search Approaches

### Approach: Multi-Vector Retrieval (Section 3)

**Implementation:** Three separate embedding vectors per video segment, combined at query time.

**Storage Architecture:**
```
Video Segment → Three 512d Embeddings:
  ├─ Visual Embedding      (visual content, scenes, actions)
  ├─ Audio Embedding       (sounds, music, ambient audio)
  └─ Transcription Embedding (spoken words, dialogue)
```

**Advantages:**
- ✅ Preserves modality-specific signal fidelity
- ✅ Transparent, modality-level debuggability
- ✅ Change weights without re-indexing
- ✅ Supports modality-specific optimization
- ✅ Foundation for adaptive architectures

**Drawbacks:**
- ❌ 3x storage footprint vs fused embeddings
- ❌ 3 vector searches instead of 1
- ❌ More complex infrastructure (3 indices)

**When to Use:**
- Production deployments requiring transparency
- Mixed query intent across modalities
- State-of-the-art semantic search
- Modality-specific tuning required

---

## 🔀 Fusion Methods

### 1. Reciprocal Rank Fusion (RRF)

**Formula:**
```
score(d) = Σ w_m / (k + rank_m(d))

Where:
  w_m = modality weight
  k = 60 (standard RRF constant)
  rank_m(d) = rank of document d in modality m
```

**Implementation:** `search_client.py:301`

**Characteristics:**
- ✅ **Robust** to score distribution differences
- ✅ **Emphasizes agreement** between modalities
- ✅ **Standard approach** (used by Elasticsearch, etc.)
- ✅ Better for **diverse query distributions**

**Default Weights:**
```python
{
  "visual": 0.8,      # 80% weight on visual ranking
  "audio": 0.1,       # 10% weight on audio ranking
  "transcription": 0.05  # 5% weight on transcription ranking
}
```

**API Usage:**
```python
results = client.search(
    query="person running in park",
    fusion_method="rrf",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.05}
)
```

---

### 2. Weighted Score Fusion

**Formula:**
```
score(s) = Σ w_m · sim(Q_m, E_m(s))

Where:
  w_m = modality weight
  sim() = cosine similarity
  Q_m = query embedding for modality m
  E_m(s) = segment embedding for modality m
```

**Implementation:** `search_client.py:359`

**Characteristics:**
- ✅ **Direct score combination**
- ✅ **Simpler** than RRF
- ⚠️ Sensitive to score distributions
- ✅ Works well with **normalized scores**

**Default Weights:**
```python
{
  "visual": 0.8,
  "audio": 0.1,
  "transcription": 0.1
}
```

**API Usage:**
```python
results = client.search(
    query="person running in park",
    fusion_method="weighted",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.1}
)
```

---

### 3. Intent-Based Dynamic Routing (Section 4.3)

**Implementation:** Uses embedding similarity to anchor prompts to automatically compute weights.

**How It Works:**
1. Pre-compute anchor embeddings for each modality (at startup)
2. For each query, compute cosine similarity to each anchor
3. Apply softmax with temperature to get normalized weights

**Formula:**
```
(w_v, w_a, w_t) = softmax(α · sim(E_query, [E_AncV, E_AncA, E_AncT]))

Where:
  α = temperature (default: 10.0)
  E_query = query embedding
  E_AncV/A/T = anchor embeddings for visual/audio/transcription
```

**Anchor Prompts:**
```python
VISUAL_ANCHOR = "What appears on screen: people, objects, scenes, actions,
                 clothing, colors, and visual composition of the video."

AUDIO_ANCHOR = "The non-speech audio in the video: music, sound effects,
                ambient sound, and other audio elements."

TRANSCRIPTION_ANCHOR = "The spoken words in the video: dialogue, narration,
                        speech, and what people say."
```

**Implementation:** `search_client.py:136-184`

**Characteristics:**
- ✅ **Query-adaptive** - weights change per query
- ✅ **Deterministic** - same query = same weights
- ✅ **Explainable** - can inspect anchor similarities
- ✅ **No training required** - uses embedding space directly
- ✅ **Fast iteration** - update anchors without retraining

**API Usage:**
```python
response = client.search_dynamic(
    query="explosion with loud bang",
    temperature=10.0  # Higher = more uniform, lower = more decisive
)

print(f"Computed weights: {response['weights']}")
# Output: {"visual": 0.45, "audio": 0.42, "transcription": 0.13}

print(f"Anchor similarities: {response['similarities']}")
# Output: {"visual": 0.78, "audio": 0.75, "transcription": 0.45}
```

**Temperature Effects:**
| Temperature | Behavior | Example Weights (visual, audio, transcription) |
|-------------|----------|-----------------------------------------------|
| `α = 1.0` | Very decisive (sharp distribution) | 0.89, 0.08, 0.03 |
| `α = 10.0` (default) | Balanced adaptation | 0.45, 0.42, 0.13 |
| `α = 50.0` | Uniform (ignores differences) | 0.34, 0.33, 0.33 |

---

## 🧠 LLM Query Decomposition (Section 3.2.2)

**Purpose:** Decompose complex natural language queries into modality-specific sub-queries for enhanced precision.

**Implementation:** `bedrock_client.py:256-401`

**How It Works:**
1. User provides a natural language query
2. Claude 3 Haiku decomposes it into three distinct queries:
   - **Visual query**: What appears on screen
   - **Audio query**: Non-speech sounds only
   - **Transcription query**: Spoken words and dialogue
3. Each sub-query gets its own embedding
4. Separate vector searches per modality using appropriate embeddings

**Example:**

**Input Query:**
```
"Ross says I take thee Rachel at a wedding"
```

**LLM Decomposition:**
```python
{
  "visual": "Ross at a wedding ceremony, wedding altar, formal attire",
  "audio": "wedding music, ceremony sounds, emotional atmosphere",
  "transcription": "Ross says I take thee Rachel"
}
```

**Model Configuration:**
- **Model:** Claude 3 Haiku (`anthropic.claude-3-haiku-20240307-v1:0`)
- **Temperature:** 0.3 (low for deterministic structured output)
- **Max Tokens:** 500

**API Usage:**
```python
# Enable decomposition with flag
results = client.search(
    query="Ross says I take thee Rachel at a wedding",
    fusion_method="rrf",
    decomposed_queries=client.bedrock.decompose_query(query)
)
```

**Web UI:** Enable "Use LLM Decomposition" toggle

**Characteristics:**
- ✅ **Precision boost** for complex multi-modal queries
- ✅ **Extracts distinct signals** from ambiguous queries
- ✅ **Context-aware expansion** - infers relevant elements
- ⚠️ **Adds latency** (~500ms for LLM call)
- ⚠️ **Requires Bedrock access** to Claude models

**Best For:**
- Complex queries spanning multiple modalities
- Queries where visual/audio/speech elements are intertwined
- When maximum precision is more important than latency

**Not Recommended For:**
- Simple single-modality queries ("red car")
- High-throughput/low-latency requirements
- Cost-sensitive applications (adds LLM inference cost)

---

## ⚖️ Modality Weight Configurations

### 1. Fixed Weights (Section 4.1)

**Method:** Manually set or statistically optimized weights applied to all queries.

**Default (Visual-Heavy):**
```python
VISUAL_WEIGHT = 0.8
AUDIO_WEIGHT = 0.1
TRANSCRIPTION_WEIGHT = 0.1
```

**Recommended Configurations by Use Case:**

| Use Case | Visual | Audio | Transcription | Example Query |
|----------|--------|-------|---------------|---------------|
| **Visual-Centric** | 0.80 | 0.10 | 0.10 | "person running", "red car crash" |
| **Dialogue-Focused** | 0.20 | 0.10 | 0.70 | "what did they say about revenue", "find where he mentions the deadline" |
| **Audio Events** | 0.30 | 0.60 | 0.10 | "explosion sound", "alarm ringing", "music playing" |
| **Balanced** | 0.40 | 0.30 | 0.30 | "wedding ceremony", "basketball game" |
| **Speech-Heavy + Visual** | 0.40 | 0.10 | 0.50 | "presenter showing slides", "interview about product" |

**Configuration Methods:**

**1. Environment Variables:**
```bash
export WEIGHT_VISUAL=0.8
export WEIGHT_AUDIO=0.1
export WEIGHT_TRANSCRIPTION=0.1
```

**2. API Parameters:**
```python
results = client.search(
    query="person laughing at joke",
    weights={"visual": 0.4, "audio": 0.3, "transcription": 0.3}
)
```

**3. Web UI Sliders:**
- Adjust visual/audio/transcription sliders in real-time
- Weights automatically normalize to sum to 1.0

**Statistical Optimization (Advanced):**

If you have historical query data with ground truth relevance labels:

```python
from search_optimization import optimize_weights

# Your evaluation dataset
eval_queries = [
    {"query": "person running", "relevant_segments": [...]},
    {"query": "alarm sound", "relevant_segments": [...]},
    # ... more examples
]

# Run grid search or Bayesian optimization
optimal_weights = optimize_weights(
    eval_queries=eval_queries,
    metric="precision@10",  # or "recall@20", "map", etc.
    search_space={
        "visual": (0.1, 0.9),
        "audio": (0.05, 0.5),
        "transcription": (0.05, 0.7)
    }
)

print(optimal_weights)
# Output: {"visual": 0.72, "audio": 0.13, "transcription": 0.15}
```

**Characteristics:**
- ✅ **Simple** - no ML training required
- ✅ **Predictable** - same weights for all queries
- ✅ **Fast** - no per-query computation
- ⚠️ **Not adaptive** - can't adjust to query intent
- ⚠️ **Requires domain knowledge** or labeled data for optimization

---

### 2. Dynamic Routing with Anchors (Section 4.3)

**Method:** Automatically compute weights per query using anchor similarity.

See [Intent-Based Dynamic Routing](#3-intent-based-dynamic-routing-section-43) above for detailed explanation.

**Query-Specific Weight Examples:**

| Query | Visual | Audio | Transcription | Reasoning |
|-------|--------|-------|---------------|-----------|
| "person running in park" | 0.71 | 0.15 | 0.14 | Strong visual signal |
| "explosion with loud bang" | 0.45 | 0.42 | 0.13 | Visual + audio balanced |
| "he says I take thee Rachel" | 0.22 | 0.12 | 0.66 | Heavily speech-focused |
| "wedding ceremony music" | 0.38 | 0.47 | 0.15 | Audio-dominant |
| "red car crash" | 0.68 | 0.18 | 0.14 | Visual with some audio |

**API Usage:**
```python
response = client.search_dynamic(
    query="explosion with loud bang",
    temperature=10.0,
    limit=50
)

# Inspect computed weights
print(f"Query: {query}")
print(f"Visual weight: {response['weights']['visual']:.2f}")
print(f"Audio weight: {response['weights']['audio']:.2f}")
print(f"Transcription weight: {response['weights']['transcription']:.2f}")

# Results
for result in response['results']:
    print(f"Segment {result['segment_id']}: {result['fusion_score']:.3f}")
```

---

## 🏗️ Architecture Overview

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
└─────────────────┘     │  │  + Multi   │  │
                        │  │    Fusion  │  │
                        │  │  + Dynamic │  │
                        │  │    Routing │  │
                        │  └────────────┘  │
                        │                  │
                        │  Fusion Methods: │
                        │  - RRF           │
                        │  - Weighted      │
                        │  - Dynamic       │
                        │                  │
                        │  Query Modes:    │
                        │  - LLM Decomp    │
                        │  - Single Query  │
                        └──────────────────┘
```

---

## 🖥️ Search UI Features

The web interface provides comprehensive search capabilities:

### Search Modes

**Multi-Vector Fusion:**
- **RRF** - Reciprocal Rank Fusion (rank-based, most robust)
- **Weighted** - Score-based fusion with adjustable weights
- **Dynamic** - Intent-based routing with automatic weight calculation

**Single Modality:**
- **Visual** - Visual content only (scenes, actions, objects)
- **Audio** - Audio/sound only (music, sound effects, ambient)
- **Speech** - Transcription/dialogue only (spoken words)

### Query Options

- **LLM Decomposition** - Enable/disable query decomposition with Claude
- **Modality Weights** - Real-time sliders for visual/audio/transcription weights
- **Temperature Control** - Adjust softmax temperature for dynamic routing (1-50)

### Backend Toggle

- **MongoDB (Single Index)** - One collection with modality filter (default)
- **S3 Vectors (Multi-Index)** - Separate index per modality

### Result Card Layout

Each search result displays comprehensive match information:

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

---

## 📁 Project Structure

```
multi-modal-video-search/
├── app.py                        # FastAPI web application (search API)
├── src/
│   ├── lambda_function.py        # Lambda handler for video processing
│   ├── bedrock_client.py         # Bedrock Marengo client + LLM decomposition
│   ├── mongodb_client.py         # MongoDB embedding storage
│   ├── s3_vectors_client.py      # S3 Vectors embedding storage & search
│   ├── search_client.py          # Multi-vector search with all fusion methods
│   └── query_fusion.py           # Legacy query fusion script
├── static/
│   └── index.html                # Search UI frontend
├── scripts/
│   ├── deploy.sh                 # AWS CLI deployment script
│   ├── mongodb_setup.md          # MongoDB Atlas setup guide
│   └── migrate_to_s3_vectors.py  # Migration script: MongoDB → S3 Vectors
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

---

## 🚀 Quick Start

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

### 3. Deploy Lambda Function

```bash
# Set MongoDB URI
export MONGODB_URI="your_mongodb_connection_string_here"

# Deploy
./scripts/deploy.sh
```

### 4. Run Search API Locally

```bash
# Start the FastAPI server
python app.py

# Open browser to http://localhost:8000
```

### 5. Process a Video

```bash
# Invoke Lambda
aws lambda invoke \
  --function-name video-embedding-pipeline \
  --region us-east-1 \
  --payload '{"s3_key": "WBD_project/Videos/Ready/sample.mp4", "bucket": "tl-brice-media"}' \
  --cli-binary-format raw-in-base64-out \
  response.json
```

### 6. Search Videos

**Via Web UI:** http://localhost:8000

**Via API:**
```bash
# Simple search with RRF fusion
curl "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "person running in park",
    "fusion_method": "rrf",
    "limit": 10
  }'

# Dynamic routing search
curl "http://localhost:8000/api/search/dynamic" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "explosion with loud bang",
    "temperature": 10.0,
    "limit": 10
  }'

# With LLM decomposition
curl "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Ross says I take thee Rachel at a wedding",
    "use_decomposition": true,
    "fusion_method": "rrf",
    "limit": 10
  }'
```

---

## 📊 MongoDB Schema

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

## 🧪 API Reference

### VideoSearchClient (search_client.py)

```python
from src.search_client import VideoSearchClient

client = VideoSearchClient(
    mongodb_uri="mongodb+srv://...",
    database_name="video_search"
)

# ============ RRF Fusion Search ============
results = client.search(
    query="person running",
    fusion_method="rrf",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.1},
    limit=10
)

# ============ Weighted Fusion Search ============
results = client.search(
    query="person running",
    fusion_method="weighted",
    weights={"visual": 0.8, "audio": 0.1, "transcription": 0.1},
    limit=10
)

# ============ Dynamic Intent Routing ============
response = client.search_dynamic(
    query="explosion with loud bang",
    temperature=10.0,
    limit=10
)
print(f"Computed weights: {response['weights']}")
print(f"Anchor similarities: {response['similarities']}")

# ============ With LLM Query Decomposition ============
decomposed = client.bedrock.decompose_query("Ross says I take thee Rachel at a wedding")
print(f"Visual: {decomposed['visual']}")
print(f"Audio: {decomposed['audio']}")
print(f"Transcription: {decomposed['transcription']}")

results = client.search(
    query="Ross says I take thee Rachel at a wedding",
    decomposed_queries=decomposed,
    fusion_method="rrf",
    limit=10
)

# ============ Single Modality Search ============
results = client.search(
    query="person running",
    modalities=["visual"],  # Only search visual modality
    limit=10
)
```

### BedrockMarengoClient (bedrock_client.py)

```python
from src.bedrock_client import BedrockMarengoClient

client = BedrockMarengoClient(region="us-east-1")

# ============ Generate Video Embeddings ============
result = client.get_video_embeddings(
    bucket="tl-brice-media",
    s3_key="WBD_project/Videos/file.mp4",
    embedding_types=["visual", "audio", "transcription"]
)

# ============ Generate Query Embedding ============
query_result = client.get_text_query_embedding("a car driving fast")

# ============ LLM Query Decomposition ============
decomposed = client.decompose_query("Ross says I take thee Rachel at a wedding")
print(decomposed)
# {
#   "original_query": "Ross says I take thee Rachel at a wedding",
#   "visual": "Ross at a wedding ceremony, wedding altar, formal attire",
#   "audio": "wedding music, ceremony sounds, emotional atmosphere",
#   "transcription": "Ross says I take thee Rachel"
# }
```

---

## 💰 Cost Estimation

Based on Marengo 3.0 pricing:

| Component | Price | Notes |
|-----------|-------|-------|
| Video embedding | $0.0007/second | For video processing |
| Text query embedding | Included | No additional cost |
| LLM decomposition (optional) | ~$0.0001/query | Claude 3 Haiku (500 tokens) |

**Example Costs:**
- **1 hour video processing:** 3,600 sec × $0.0007 = **$2.52**
- **1,000 searches (no decomposition):** **$0** (text embeddings included)
- **1,000 searches (with decomposition):** ~**$0.10** (LLM calls)

---

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGODB_URI` | Required | MongoDB connection string |
| `MONGODB_DATABASE` | `video_search` | Database name |
| `AWS_REGION` | `us-east-1` | AWS region for Bedrock |
| `S3_BUCKET` | `tl-brice-media` | S3 bucket for videos |
| `CLOUDFRONT_DOMAIN` | `d2h48upmn4e6uy.cloudfront.net` | CloudFront domain |
| `WEIGHT_VISUAL` | `0.8` | Default visual weight (fixed mode) |
| `WEIGHT_AUDIO` | `0.1` | Default audio weight (fixed mode) |
| `WEIGHT_TRANSCRIPTION` | `0.1` | Default transcription weight (fixed mode) |

---

## 🐛 Troubleshooting

### Lambda Timeout

- Default timeout is 15 minutes (900 seconds)
- For very long videos (>2 hours), consider splitting into segments
- Increase memory to 2048MB or higher for faster processing

### Vector Search Returns No Results

1. Verify index is in **Active** state in Atlas UI
2. Check embedding dimensions match (512)
3. Ensure collection has documents
4. Verify filter field values match exactly

### LLM Decomposition Fails

1. Verify Bedrock access to Claude 3 Haiku model
2. Check model ID is correct: `anthropic.claude-3-haiku-20240307-v1:0`
3. Ensure AWS credentials have `bedrock:InvokeModel` permission
4. Check CloudWatch logs for detailed error messages

### Connection Errors

1. Verify MongoDB Atlas IP whitelist includes Lambda/App Runner IPs
2. Check connection string format
3. For testing, use 0.0.0.0/0 in Atlas Network Access

---

## 📚 References

- [TwelveLabs Multi-Vector Guidance](./A%20Guidance%20on%20Multi-Vector%20Video%20Search%20with%20TwelveLabs%20Marengo.pdf) - Complete whitepaper
- [MongoDB Atlas Vector Search](https://www.mongodb.com/docs/atlas/atlas-vector-search/)
- [Amazon S3 Vectors Documentation](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
- [AWS Bedrock Marengo](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html)
- [Claude 3 Models](https://docs.anthropic.com/claude/docs/models-overview)

---

## 📝 License

Internal use only. All rights reserved.
