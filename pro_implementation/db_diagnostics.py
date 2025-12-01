#!/usr/bin/env python3
"""
DB Diagnostics Script for Ayush RAG Pipeline

This script checks:
- Which DB folders exist (preprocessed_db / vector_db)
- Whether Chroma collections exist
- Document counts for each collection
- Sample metadata from stored chunks
"""

from chromadb import PersistentClient
from pathlib import Path
import os

BASE = Path(__file__).resolve().parent.parent

DB_FOLDERS = ["preprocessed_db", "vector_db", "chroma", "db"]


def check_db_folder(path: Path):
    print("\n🔍 Checking folder:", path)

    if not path.exists():
        print(" ❌ Folder does NOT exist.")
        return False

    # Check if Chroma files exist
    sqlite_file = path / "chroma.sqlite3"

    if sqlite_file.exists():
        print(" ✅ Found chroma.sqlite3")
    else:
        print(" ⚠️ WARNING: No chroma.sqlite3 found (DB may be empty).")

    return True


def inspect_chroma(path: Path):
    try:
        client = PersistentClient(path=str(path))
    except Exception as e:
        print(" ❌ Failed to load DB:", e)
        return

    print("\n📁 Listing Collections:")
    try:
        collections = client.list_collections()
    except Exception:
        print(" ❌ Could not list collections — DB may be corrupted.")
        return

    if not collections:
        print(" ⚠️ No collections found in this DB.")
        return

    for col in collections:
        print(f"\n🗂 Collection Name: {col.name}")
        collection = client.get_collection(col.name)

        try:
            count = collection.count()
        except Exception:
            print("   ❌ Failed to count documents (possible corruption).")
            continue

        print(f"   📦 Documents: {count}")

        # Fetch sample docs
        if count > 0:
            try:
                sample = collection.peek()
                print("   📝 Sample document metadata:")
                print("    ", sample["metadatas"][0])
            except Exception:
                print("   ⚠️ Could not load metadata sample.")


def main():
    print("=====================================")
    print("     🧪 AYUSH RAG DB DIAGNOSTICS")
    print("=====================================")

    for folder in DB_FOLDERS:
        db_path = BASE / folder
        check = check_db_folder(db_path)

        if check:
            inspect_chroma(db_path)

    print("\n🎉 Diagnostics complete!")


if __name__ == "__main__":
    main()
