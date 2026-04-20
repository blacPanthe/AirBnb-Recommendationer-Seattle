# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**First-time setup — train models (takes ~5–10 min):**
```bash
python3 preprocess.py
```
Must be re-run whenever `listings.csv`, `reviews.csv`, or BERTopic/UMAP/HDBSCAN parameters change. Outputs to `models/`.

**Start the web app:**
```bash
python3 app.py
# Open http://localhost:8080
```
Port 5000 is blocked by macOS AirPlay — always use 8080.

**Install dependencies:**
```bash
pip3 install flask bertopic sentence-transformers umap-learn hdbscan nltk scikit-learn numpy pandas certifi
```

**NLTK data** is auto-downloaded by `preprocess.py` on first run. If SSL errors occur on macOS, run `/Applications/Python\ 3.X/Install\ Certificates.command` first.

## Architecture

Two-stage pipeline: a one-time preprocessing step produces saved artifacts, which the Flask app loads at startup.

### Stage 1 — `preprocess.py`
Reads `listings.csv` + `reviews.csv` → cleans text → trains BERTopic → writes `models/`:

| Artifact | Contents |
|---|---|
| `bertopic_model/` | Saved BERTopic (pickle serialization) |
| `listings_enriched.pkl` | DataFrame: listing metadata + `dominant_topic`, `topic_label`, `avg_sentiment`, `sentiment_label` |
| `similarity_matrix.npy` | Float32 cosine similarity matrix `(n_listings × n_listings)` over topic-prob + sentiment features |
| `topic_labels.json` | `{label_map: {topic_id: label}, keywords: {label: [words]}}` |
| `id_to_idx.json` | `{listing_id: row_index}` for O(1) similarity lookups |

**Key preprocessing decisions:**
- Reviews are aggregated per listing before embedding (one document per listing, not per review)
- POS-tagging runs on original-case text before lowercasing to correctly strip proper nouns (NNP/NNPS) — this removes host names that would otherwise dominate BERTopic clusters
- `CountVectorizer(min_df=10)` further suppresses rare name tokens from topic representations
- `nr_topics=15` merges HDBSCAN's raw clusters down to ~14 meaningful topics
- VADER sentiment is computed per review then averaged per listing

### Stage 2 — `recommender.py`
Loads all `models/` artifacts once at module import (lazy, cached). Exposes three functions consumed by Flask:
- `get_topics()` — topic list with listing counts and keywords
- `get_listings_by_topic(topic_id, top_n)` — listings filtered by dominant topic, sorted by rating
- `get_similar_listings(listing_id, top_n)` — cosine similarity lookup via `similarity_matrix.npy`
- `search_listings_by_name(query)` — substring match for autocomplete

### Stage 3 — `app.py` + frontend
Flask serves four API routes (`/api/topics`, `/api/listings`, `/api/recommend/similar`, `/api/search`) plus `templates/index.html`. All frontend logic is in `static/app.js` (vanilla JS, no build step). Styling in `static/style.css`.

The UI is a single-page Topic Browser: sidebar of topic pills with hover tooltips showing BERTopic keywords; clicking a topic fetches and renders listing cards.

## Data

`listings.csv` (6,862 rows, 79 cols) and `reviews.csv` (541,092 rows) are the raw Seattle Airbnb dataset. Only English reviews (ASCII-dominant filter) are used. After aggregation, ~5,926 listings have sufficient review text for topic modeling.
