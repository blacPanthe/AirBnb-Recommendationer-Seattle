"""
One-time preprocessing script.
Run: python preprocess.py
Outputs models/ directory with all artifacts needed by the Flask app.
"""

import os
import re
import json
import pickle
import ssl
import warnings
warnings.filterwarnings("ignore")

# Fix macOS Python SSL certificate verification for NLTK downloads
try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
except Exception:
    ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

# ── NLTK downloads ──────────────────────────────────────────────────────────
for pkg in ["punkt", "stopwords", "wordnet", "vader_lexicon", "omw-1.4", "punkt_tab", "averaged_perceptron_tagger_eng"]:
    nltk.download(pkg, quiet=True)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

LISTINGS_CSV = os.path.join(os.path.dirname(__file__), "listings.csv")
REVIEWS_CSV  = os.path.join(os.path.dirname(__file__), "reviews.csv")

def make_topic_label(keywords):
    """Turn a list of top keywords into a readable topic label."""
    if not keywords:
        return "General Stay"
    # Title-case the top 3 keywords and join
    top = [w.title() for w in keywords[:3]]
    return " · ".join(top)


# ── 1. Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
listings_df = pd.read_csv(LISTINGS_CSV, low_memory=False)
reviews_df  = pd.read_csv(REVIEWS_CSV,  low_memory=False)

print(f"  Listings: {len(listings_df):,} rows")
print(f"  Reviews : {len(reviews_df):,} rows")


# ── 2. Clean listings ────────────────────────────────────────────────────────
print("Cleaning listings...")
keep_cols = [
    "id", "name", "description", "room_type", "bedrooms", "bathrooms",
    "price", "latitude", "longitude",
    "review_scores_rating", "review_scores_cleanliness",
    "review_scores_communication", "review_scores_location",
    "neighbourhood_cleansed", "property_type", "accommodates",
]
keep_cols = [c for c in keep_cols if c in listings_df.columns]
listings_df = listings_df[keep_cols].copy()
listings_df.rename(columns={"id": "listing_id"}, inplace=True)

# Clean price
if "price" in listings_df.columns:
    listings_df["price"] = (
        listings_df["price"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )

listings_df["listing_id"] = pd.to_numeric(listings_df["listing_id"], errors="coerce")
listings_df.dropna(subset=["listing_id"], inplace=True)
listings_df["listing_id"] = listings_df["listing_id"].astype(int)


# ── 3. Clean & filter reviews ────────────────────────────────────────────────
print("Cleaning reviews...")
reviews_df.dropna(subset=["comments"], inplace=True)
reviews_df["comments"] = reviews_df["comments"].astype(str)
reviews_df = reviews_df[reviews_df["comments"].str.len() >= 20].copy()

# Language filter — keep only ASCII-dominant reviews (fast proxy for English)
def is_mostly_ascii(text, threshold=0.85):
    if not text:
        return False
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return (ascii_count / len(text)) >= threshold

reviews_df = reviews_df[reviews_df["comments"].apply(is_mostly_ascii)].copy()
print(f"  Reviews after language filter: {len(reviews_df):,}")


# ── 4. Text preprocessing ────────────────────────────────────────────────────
print("Preprocessing text...")
stop_words = set(stopwords.words("english"))
custom_stop = {
    "airbnb", "stay", "place", "host", "listing", "room", "apartment",
    "house", "home", "would", "also", "really", "great", "good",
    "well", "definitely", "highly", "recommend", "everything",
    # common host/reviewer first names that skew BERTopic clusters
    "michael", "john", "james", "david", "daniel", "ryan", "kevin", "brian",
    "chris", "mark", "paul", "jason", "matt", "alex", "andrew", "josh",
    "sarah", "emily", "jessica", "jennifer", "ashley", "amanda", "rachel",
    "lauren", "emma", "lisa", "amy", "kelly", "kim", "kate", "anna",
    "liz", "marti", "vince", "evan", "eric", "adam", "tyler", "nathan",
    "bill", "bob", "ben", "sam", "tom", "tim", "mike", "nick", "dan",
    "eva", "evalyne", "melchor", "johan", "kiera", "jan", "anthony",
    "margaret", "patricia", "linda", "barbara", "deborah", "sharon",
    "thank", "thanks", "us", "one", "get", "got", "said", "told",
    "made", "make", "like", "just", "very", "nice", "wonderful", "perfect",
    "excellent", "amazing", "fantastic", "awesome", "lovely", "beautiful",
}
stop_words |= custom_stop
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    # POS-tag on original case first to correctly identify proper nouns
    raw_tokens = word_tokenize(re.sub(r"http\S+|www\S+", "", text))
    tagged = nltk.pos_tag(raw_tokens)
    # Drop proper nouns (names) before any other processing
    filtered = [w for w, tag in tagged if tag not in ("NNP", "NNPS")]
    # Now lowercase and clean
    text = clean_text(" ".join(filtered))
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)

reviews_df["clean_comments"] = reviews_df["comments"].apply(preprocess)
reviews_df = reviews_df[reviews_df["clean_comments"].str.len() > 0].copy()


# ── 5. Aggregate reviews per listing ─────────────────────────────────────────
print("Aggregating reviews per listing...")
reviews_df["listing_id"] = pd.to_numeric(reviews_df["listing_id"], errors="coerce")
reviews_df.dropna(subset=["listing_id"], inplace=True)
reviews_df["listing_id"] = reviews_df["listing_id"].astype(int)

aggregated = (
    reviews_df.groupby("listing_id")["clean_comments"]
    .apply(lambda x: " ".join(x))
    .reset_index()
    .rename(columns={"clean_comments": "aggregated_text"})
)

# Keep only listings that appear in listings_df
aggregated = aggregated[aggregated["listing_id"].isin(listings_df["listing_id"])].copy()
print(f"  Listings with reviews: {len(aggregated):,}")


# ── 6. Sentiment analysis ─────────────────────────────────────────────────────
print("Running sentiment analysis...")
sid = SentimentIntensityAnalyzer()

# Per-review sentiment
reviews_df["sentiment_compound"] = reviews_df["comments"].apply(
    lambda t: sid.polarity_scores(t)["compound"]
)

sentiment_per_listing = (
    reviews_df.groupby("listing_id")["sentiment_compound"]
    .mean()
    .reset_index()
    .rename(columns={"sentiment_compound": "avg_sentiment"})
)

def sentiment_label(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    return "Neutral"

sentiment_per_listing["sentiment_label"] = sentiment_per_listing["avg_sentiment"].apply(sentiment_label)


# ── 7. BERTopic ───────────────────────────────────────────────────────────────
print("Training BERTopic (this may take a few minutes)...")

docs = aggregated["aggregated_text"].tolist()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

umap_model = UMAP(
    n_neighbors=10,
    n_components=10,
    min_dist=0.0,
    metric="cosine",
    random_state=42,
)

hdbscan_model = HDBSCAN(
    min_cluster_size=8,
    min_samples=3,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True,
)

representation_model = KeyBERTInspired()

# min_df=10: ignore words that appear in fewer than 10 docs (filters rare host names)
vectorizer_model = CountVectorizer(min_df=10, ngram_range=(1, 2))

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    representation_model=representation_model,
    vectorizer_model=vectorizer_model,
    nr_topics=15,
    calculate_probabilities=True,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs)

# Reduce outliers — assign -1 docs to their nearest topic
print("Reducing outliers...")
topics = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities")
topic_model.update_topics(docs, topics=topics)

print(f"  Topics found: {len(set(topics))}")


# ── 8. Assign topic labels ────────────────────────────────────────────────────
print("Assigning topic labels...")
unique_topics = sorted(t for t in set(topics) if t != -1)

# Auto-generate labels from BERTopic's top keywords for each topic
topic_label_map = {}
for tid in unique_topics:
    try:
        words = topic_model.get_topic(tid)
        kws = [w for w, _ in words[:3]] if words else []
    except Exception:
        kws = []
    topic_label_map[tid] = make_topic_label(kws)

topic_label_map[-1] = "Uncategorized"

aggregated["dominant_topic"] = topics
aggregated["topic_label"] = [topic_label_map.get(t, "Uncategorized") for t in topics]

# Topic probability vectors
try:
    topic_probs_matrix = topic_model.get_document_info(docs)["Probability"].tolist()
    # Fall back to sparse vector approach if needed
except Exception:
    topic_probs_matrix = None

# Build dense probability array
all_topic_ids = sorted(t for t in set(topics) if t != -1)
n_topics = len(all_topic_ids)
topic_idx_map = {tid: i for i, tid in enumerate(all_topic_ids)}

prob_vectors = np.zeros((len(docs), n_topics), dtype=np.float32)
for i, (t, p) in enumerate(zip(topics, probs)):
    if t != -1 and t in topic_idx_map:
        prob_vectors[i, topic_idx_map[t]] = float(p) if not hasattr(p, "__len__") else float(np.max(p))

aggregated["topic_probs"] = list(prob_vectors)


# ── 9. Build enriched listings DataFrame ─────────────────────────────────────
print("Building enriched listings DataFrame...")
enriched = listings_df.merge(aggregated[["listing_id", "dominant_topic", "topic_label", "topic_probs"]], on="listing_id", how="inner")
enriched = enriched.merge(sentiment_per_listing, on="listing_id", how="left")
enriched["avg_sentiment"].fillna(0.0, inplace=True)
enriched["sentiment_label"].fillna("Neutral", inplace=True)

# Ensure review_scores_rating is numeric
if "review_scores_rating" in enriched.columns:
    enriched["review_scores_rating"] = pd.to_numeric(enriched["review_scores_rating"], errors="coerce")

print(f"  Enriched listings: {len(enriched):,}")


# ── 10. Similarity matrix ─────────────────────────────────────────────────────
print("Computing similarity matrix...")
prob_matrix = np.vstack(enriched["topic_probs"].values)
sentiment_col = enriched["avg_sentiment"].values.reshape(-1, 1)

# Normalize sentiment to [0,1] range before concatenating
sentiment_norm = (sentiment_col - sentiment_col.min()) / (sentiment_col.max() - sentiment_col.min() + 1e-9)

features = np.hstack([prob_matrix, sentiment_norm])
features_normalized = normalize(features, norm="l2")

sim_matrix = cosine_similarity(features_normalized).astype(np.float32)
print(f"  Similarity matrix shape: {sim_matrix.shape}")


# ── 11. Extract topic keywords for display ────────────────────────────────────
print("Extracting topic keywords...")
topic_keywords = {}
for tid in unique_topics:
    try:
        words = topic_model.get_topic(tid)
        if words:
            kws = [w for w, _ in words[:6]]
            topic_keywords[topic_label_map[tid]] = kws
    except Exception:
        topic_keywords[topic_label_map.get(tid, str(tid))] = []


# ── 12. Save all artifacts ────────────────────────────────────────────────────
print("Saving artifacts...")

# BERTopic model
bertopic_path = os.path.join(MODELS_DIR, "bertopic_model")
topic_model.save(bertopic_path, serialization="pickle", save_ctfidf=True)

# Enriched listings (drop topic_probs array column — stored separately)
enriched_save = enriched.drop(columns=["topic_probs"])
with open(os.path.join(MODELS_DIR, "listings_enriched.pkl"), "wb") as f:
    pickle.dump(enriched_save, f)

# Probability matrix (for similarity lookups)
np.save(os.path.join(MODELS_DIR, "similarity_matrix.npy"), sim_matrix)

# Topic labels JSON
with open(os.path.join(MODELS_DIR, "topic_labels.json"), "w") as f:
    json.dump({
        "label_map": {str(k): v for k, v in topic_label_map.items()},
        "keywords":  topic_keywords,
    }, f, indent=2)

# Listing ID → row index map
id_to_idx = {int(lid): int(i) for i, lid in enumerate(enriched["listing_id"].values)}
with open(os.path.join(MODELS_DIR, "id_to_idx.json"), "w") as f:
    json.dump(id_to_idx, f)

print("\nDone! Artifacts saved to models/")
print("  models/bertopic_model/")
print("  models/listings_enriched.pkl")
print("  models/similarity_matrix.npy")
print("  models/topic_labels.json")
print("  models/id_to_idx.json")
print("\nNext: python app.py")
