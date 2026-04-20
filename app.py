"""
Flask web server for the Airbnb Recommendation System.
Run: python app.py
Then open: http://localhost:5000
"""

from flask import Flask, jsonify, render_template, request
import recommender

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/topics")
def topics():
    try:
        return jsonify(recommender.get_topics())
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 503


@app.route("/api/listings")
def listings():
    try:
        topic_id = request.args.get("topic", type=int)
        top_n    = request.args.get("top_n", default=20, type=int)
        if topic_id is None:
            return jsonify({"error": "topic parameter required"}), 400
        return jsonify(recommender.get_listings_by_topic(topic_id, top_n))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommend/similar")
def similar():
    try:
        listing_id = request.args.get("listing_id", type=int)
        top_n      = request.args.get("top_n", default=5, type=int)
        if listing_id is None:
            return jsonify({"error": "listing_id parameter required"}), 400
        return jsonify(recommender.get_similar_listings(listing_id, top_n))
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search")
def search():
    try:
        q     = request.args.get("q", "").strip()
        limit = request.args.get("limit", default=10, type=int)
        if not q:
            return jsonify([])
        return jsonify(recommender.search_listings_by_name(q, limit))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Starting Airbnb Recommendation System...")
    print("Open http://localhost:8080 in your browser")
    app.run(debug=True, port=8080)
