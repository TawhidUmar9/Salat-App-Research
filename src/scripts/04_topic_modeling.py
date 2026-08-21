"""
Phase 3 — Topic Modelling
==========================
BERTopic with sentence-transformers on the analysis sub-corpus.

Input:
    - data/master_reviews_with_sentiment.parquet (C_en subset)

Output:
    - data/topics.parquet (reviewId, topic_id, topic_label, topic_probability)

See implementation_plan.md §7 for full specification.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "master_reviews_with_sentiment.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "topics.parquet"

# Model for embeddings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternative for multilingual run over C_trans:
# EMBEDDING_MODEL_MULTI = "paraphrase-multilingual-MiniLM-L12-v2"


def build_topic_model(docs: list[str]):
    """
    TODO:
    - Initialize BERTopic with:
        - SentenceTransformer(EMBEDDING_MODEL)
        - min_topic_size tuned to corpus size
        - Custom CountVectorizer with Islamic-domain stopword list:
          app names, "allah", "islam", "muslim", "app", "good", "nice"
          (otherwise these dominate every cluster)
        - nr_topics="auto", then manual reduction to 15–25 interpretable themes
    - Run on C_en only (C_text with lang == 'en')
    - DO NOT run on unfiltered corpus — 50-60% under 5 words produces
      a giant "good app" cluster and nothing useful
    """
    pass


def label_topics(topic_model, docs: list[str]) -> pd.DataFrame:
    """
    TODO:
    - Label topics from top-20 representative documents
    - Tag each topic with relevant RQ(s)
    - Compute prevalence per app
    - Generate topic × sentiment heatmap data
    """
    pass


def run_subtopic_models(df: pd.DataFrame):
    """
    TODO:
    - Sub-topic model within tracker-aspect reviews (for RQ1):
      streak-anxiety vs. streak-motivation resolution
    - Sub-topic model within women/privacy reviews (for RQ4)
    - The ecosystem-level model won't resolve these fine-grained distinctions
    """
    pass


def main():
    """
    Topic modelling pipeline:
    1. Load C_en reviews
    2. Build main BERTopic model
    3. Label and tag topics
    4. Run sub-topic models for RQ1, RQ4
    5. Save topics.parquet
    """
    # TODO: Wire up pipeline
    pass


if __name__ == "__main__":
    main()
