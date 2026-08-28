import os
import json
import uuid

SOURCES_DIR = "sources"


def add_uuids_to_file(filepath):
    """
    Add a UUID only to insights that do not already have one.

    Existing UUIDs are preserved so records keep stable identifiers
    across repeated ingestion runs.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)

    if "insights" not in data:
        return

    updated = False

    for insight in data["insights"]:
        if not insight.get("uuid"):
            insight["uuid"] = str(uuid.uuid4())
            updated = True

    if updated:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Added missing UUIDs: {filepath}")
    else:
        print(f"No UUID changes needed: {filepath}")


def process_all_sources():
    for root, _, files in os.walk(SOURCES_DIR):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                add_uuids_to_file(filepath)


if __name__ == "__main__":
    process_all_sources()
