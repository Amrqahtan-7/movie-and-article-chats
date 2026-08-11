"""
ONE-TIME cleanup script.

Purpose: delete every chunk in the Qdrant 'movies_demo' collection that came
from a user file upload (tagged "source": "upload" at upload time), while
leaving the original movie dataset (wiki_movie_plots_deduped.csv) untouched.

Run this ONCE, before switching to the new per-user server.py, so the
chatbot starts fresh with only the base dataset. Safe to delete this file
afterward — it is not imported by server.py and has no other purpose.

Usage:
    python cleanup_uploads.py
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "movies_demo"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_index(key_path: str):
    """Qdrant requires a payload index to exist before you can filter on a
    field. This creates one if it doesn't already exist yet — safe to call
    even if the index is already there (it just prints a message and
    continues)."""
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=key_path,
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        print(f"Created index on '{key_path}'.")
    except Exception as e:
        print(f"Index step for '{key_path}': {e}")


def inspect_one_point():
    """Look at one stored point so we can confirm the real metadata key path
    before trusting it in a delete filter. Prevents silently deleting nothing
    (or worse, deleting the wrong things) due to a wrong key guess."""
    points, _ = client.scroll(collection_name=COLLECTION_NAME, limit=1, with_payload=True)
    if not points:
        print("Collection is empty or doesn't exist — nothing to inspect.")
        return None
    payload = points[0].payload
    print("Sample point payload structure:")
    print(payload)
    return payload


def count_upload_chunks(key_path: str):
    """Count how many points match source == 'upload' under the given key path,
    without deleting anything yet."""
    result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=key_path,
                    match=qdrant_models.MatchValue(value="upload"),
                )
            ]
        ),
        exact=True,
    )
    return result.count


def delete_upload_chunks(key_path: str):
    """Delete all points where key_path == 'upload'."""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key=key_path,
                        match=qdrant_models.MatchValue(value="upload"),
                    )
                ]
            )
        ),
    )


if __name__ == "__main__":
    print(f"Connecting to Qdrant collection '{COLLECTION_NAME}'...\n")

    payload = inspect_one_point()
    if payload is None:
        exit()

    # Try the most likely key path first based on how langchain_qdrant
    # typically stores metadata. Adjust this if inspect_one_point() above
    # shows a different structure (e.g. "source" instead of "metadata.source").
    key_path = "metadata.source"

    print(f"\nEnsuring index exists on '{key_path}' before filtering...")
    ensure_index(key_path)

    print(f"\nChecking how many chunks match {key_path} == 'upload' ...")
    count = count_upload_chunks(key_path)
    print(f"Found {count} chunks tagged as uploads.")

    if count == 0:
        print(
            "\nZero matches. Before assuming there's nothing to delete, double-check "
            "the sample payload printed above — the key might be nested differently "
            "(e.g. just 'source' with no 'metadata.' prefix). Update key_path in this "
            "script and re-run if needed."
        )
        exit()

    confirm = input(f"\nType 'DELETE' to permanently remove these {count} chunks: ")
    if confirm.strip() == "DELETE":
        delete_upload_chunks(key_path)
        print("Done. Upload chunks removed. Base dataset left untouched.")
    else:
        print("Cancelled. Nothing was deleted.")