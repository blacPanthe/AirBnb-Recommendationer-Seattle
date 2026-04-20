"""
Recommendation logic module.
Loads saved model artifacts once at import time (cached in memory).
"""

import os
import json
import pickle
import numpy as np

_BASE = os.path.dirname(__file__)
_MODELS = os.path.join(_BASE, "models")

# ── Lazy-loaded globals ───────────────────────────────────────────────────────
_listings_df    = None
_sim_matrix     = None
_id_to_idx      = None
_topic_meta     = None   # {label_map, keywords}


def _load():
    global _listings_df, _sim_matrix, _id_to_idx, _topic_meta

    if _listings_df is not None:
        return  # already loaded

    missing = [
        f for f in ["listings_enriched.pkl", "similarity_matrix.npy",
                     "topic_labels.json", "id_to_idx.json"]
        if not os.path.exists(os.path.join(_MODELS, f))
    ]
    if missing:
        raise RuntimeError(
            f"Model artifacts not found: {missing}\n"
            "Run: python preprocess.py"
        )

    with open(os.path.join(_MODELS, "listings_enriched.pkl"), "rb") as f:
        _listings_df = pickle.load(f)

    _sim_matrix = np.load(os.path.join(_MODELS, "similarity_matrix.npy"))

    with open(os.path.join(_MODELS, "id_to_idx.json")) as f:
        _id_to_idx = {int(k): v for k, v in json.load(f).items()}

    with open(os.path.join(_MODELS, "topic_labels.json")) as f:
        _topic_meta = json.load(f)


def _listing_to_dict(row):
    """Convert a DataFrame row (Series) to a JSON-safe dict."""
    def safe(val):
        if val is None:
            return None
        if isinstance(val, float) and np.isnan(val):
            return None
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return round(float(val), 2)
        return val

    return {
        "listing_id":   safe(row.get("listing_id")),
        "name":         safe(row.get("name")),
        "description":  str(row.get("description") or "")[:300],
        "room_type":    safe(row.get("room_type")),
        "bedrooms":     safe(row.get("bedrooms")),
        "price":        safe(row.get("price")),
        "rating":       safe(row.get("review_scores_rating")),
        "latitude":     safe(row.get("latitude")),
        "longitude":    safe(row.get("longitude")),
        "neighbourhood":safe(row.get("neighbourhood_cleansed")),
        "property_type":safe(row.get("property_type")),
        "accommodates": safe(row.get("accommodates")),
        "dominant_topic": safe(row.get("dominant_topic")),
        "topic_label":  safe(row.get("topic_label")),
        "sentiment":    safe(row.get("avg_sentiment")),
        "sentiment_label": safe(row.get("sentiment_label")),
        "keywords":     _topic_meta["keywords"].get(str(row.get("topic_label")), []),
    }


# ── Public API ────────────────────────────────────────────────────────────────

def get_topics():
    """Return list of {id, label, listing_count, keywords} sorted by listing count desc."""
    _load()
    counts = _listings_df.groupby(["dominant_topic", "topic_label"]).size().reset_index(name="count")
    results = []
    for _, row in counts.iterrows():
        tid   = int(row["dominant_topic"])
        label = str(row["topic_label"])
        kws   = _topic_meta["keywords"].get(label, [])
        results.append({"id": tid, "label": label, "listing_count": int(row["count"]), "keywords": kws})
    results.sort(key=lambda x: x["listing_count"], reverse=True)
    return results


def get_listings_by_topic(topic_id: int, top_n: int = 20):
    """Return top_n listings for a topic, sorted by rating descending."""
    _load()
    subset = _listings_df[_listings_df["dominant_topic"] == topic_id].copy()
    if "review_scores_rating" in subset.columns:
        subset = subset.sort_values("review_scores_rating", ascending=False, na_position="last")
    subset = subset.head(top_n)
    return [_listing_to_dict(row) for _, row in subset.iterrows()]


def get_similar_listings(listing_id: int, top_n: int = 5):
    """Return top_n listings most similar to the given listing_id."""
    _load()
    if listing_id not in _id_to_idx:
        raise ValueError(f"Listing ID {listing_id} not found in the dataset.")

    row_idx = _id_to_idx[listing_id]
    sim_scores = _sim_matrix[row_idx].copy()
    sim_scores[row_idx] = -1  # exclude self

    top_indices = np.argsort(sim_scores)[::-1][:top_n]

    results = []
    for idx in top_indices:
        listing_row = _listings_df.iloc[idx]
        d = _listing_to_dict(listing_row)
        d["similarity_score"] = round(float(sim_scores[idx]) * 100, 1)
        results.append(d)
    return results


def search_listings_by_name(query: str, limit: int = 10):
    """Return listings whose name contains the query string (case-insensitive)."""
    _load()
    mask = _listings_df["name"].astype(str).str.lower().str.contains(query.lower(), na=False)
    subset = _listings_df[mask].head(limit)
    return [_listing_to_dict(row) for _, row in subset.iterrows()]
