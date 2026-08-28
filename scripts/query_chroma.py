import chromadb
from chromadb.utils import embedding_functions


# -----------------------------
# Configuration
# -----------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "hair_care_insights"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# Load collection
# -----------------------------

def get_collection():
    embedding_function = (
        embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL_NAME
        )
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )


# -----------------------------
# Query function
# -----------------------------

def semantic_query(
    query_text,
    n_results=5,
    metadata_filter=None
):
    """
    Perform semantic search over the Fro
    hair-care knowledge base.
    """

    collection = get_collection()

    print(f"Searching for: '{query_text}'")

    query_args = {
        "query_texts": [query_text],
        "n_results": n_results,
    }

    if metadata_filter:
        query_args["where"] = metadata_filter

    return collection.query(**query_args)


# -----------------------------
# Display results
# -----------------------------

def display_results(results):
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    print("\n🔍 Top Results:\n")

    for i, (doc, meta, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):
        print(f"Result {i}")
        print("-" * 60)

        print(doc)

        print("\nMetadata:")
        for key, value in meta.items():
            print(f"  {key}: {value}")

        print(f"\nDistance: {distance:.4f}")
        print()


# -----------------------------
# Example usage
# -----------------------------

if __name__ == "__main__":

    query = (
        "Why is Afro-textured hair more prone "
        "to breakage during grooming?"
    )

    filters = {
        "evidence_strength": "High"
    }

    # For an unfiltered search:
    # filters = None

    results = semantic_query(
        query_text=query,
        n_results=5,
        metadata_filter=filters
    )

    display_results(results)
