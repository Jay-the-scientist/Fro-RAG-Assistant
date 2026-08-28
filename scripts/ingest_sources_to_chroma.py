import os
import json
import chromadb
from chromadb.utils import embedding_functions


# -----------------------------
# Configuration
# -----------------------------

SOURCES_DIR = "sources"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hair_care_insights"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------
# Helper functions
# -----------------------------

def normalize_metadata(value):
    """
    Chroma metadata values must be scalar values.

    Convert lists to comma-separated strings.
    Leave normal strings/numbers unchanged.
    """
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)

    return value


def build_embedding_text(insight):
    """
    Construct the semantic text stored and embedded in Chroma.

    Claim and mechanism contain the primary scientific meaning
    used during semantic retrieval.
    """
    claim = insight.get("claim", "")
    mechanism = insight.get("mechanism") or ""

    if mechanism:
        return f"Claim: {claim}\nMechanism: {mechanism}"

    return f"Claim: {claim}"


def clean_metadata(metadata):
    """
    Remove metadata fields whose values are None.

    Chroma does not accept None as a metadata value.
    """
    return {
        key: normalize_metadata(value)
        for key, value in metadata.items()
        if value is not None
    }


# -----------------------------
# Main ingestion function
# -----------------------------

def ingest_sources():
    print("Initializing embedding model...")

    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    print("Initializing Chroma client...")

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    for filename in os.listdir(SOURCES_DIR):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(SOURCES_DIR, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception as e:
            print(f"Skipping {filename} (JSON error): {e}")
            continue

        # -----------------------------
        # Source-level metadata
        # -----------------------------

        article_title = data.get("article_title")
        source_url = data.get("source_url")
        source_type = data.get("source_type")
        source_institution = data.get("source_institution")
        date_published = data.get("date_published")

        insights = data.get("insights", [])

        documents = []
        metadatas = []
        ids = []

        # -----------------------------
        # Process individual insights
        # -----------------------------

        for insight in insights:
            insight_uuid = insight.get("uuid")

            if not insight_uuid:
                print(
                    f"Skipping insight without UUID in {filename}. "
                    "Run add_uuids.py first."
                )
                continue

            embedding_text = build_embedding_text(insight)

            metadata = clean_metadata({
                # Source metadata
                "article_title": article_title,
                "source_url": source_url,
                "source_type": source_type,
                "source_institution": source_institution,
                "date_published": date_published,

                # Insight metadata
                "category": insight.get("category"),
                "knowledge_type": insight.get("knowledge_type"),
                "community_terms": insight.get("community_terms"),
                "scientific_mappings": insight.get(
                    "scientific_mappings"
                ),
                "applicability": insight.get("applicability"),
                "evidence_strength": insight.get(
                    "evidence_strength"
                ),
                "confidence_notes": insight.get(
                    "confidence_notes"
                ),
                "date_added": insight.get("date_added"),
            })

            documents.append(embedding_text)
            metadatas.append(metadata)
            ids.append(insight_uuid)

        if documents:
            collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            print(
                f"Ingested {len(documents)} insights "
                f"from {filename}"
            )

    print(
        "✅ Ingestion complete. "
        "Vector database persisted to disk."
    )


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    ingest_sources()
