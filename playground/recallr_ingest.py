import argparse
import glob
import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from recallrai import RecallrAI
from recallrai.models import MessageRole
from recallrai.exceptions import UserNotFoundError


def main():
    parser = argparse.ArgumentParser(description="Ingest JSON sessions into Recallr AI.")
    parser.add_argument("directory", help="The directory containing the JSON files and the .env file")
    args = parser.parse_args()

    ingest_dir = Path(args.directory).resolve()
    env_path = ingest_dir / ".env"
    
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded credentials from {env_path}")
    else:
        print(f"Warning: .env file not found at {env_path}. Relying on shell environment variables.")

    project_id = os.getenv("RECALLR_PROJECT_ID")
    api_key = os.getenv("RECALLR_API_KEY")
    user_id = os.getenv("RECALLR_USER_ID")

    if not all([project_id, api_key, user_id]):
        raise SystemExit("Error: RECALLR_PROJECT_ID, RECALLR_API_KEY, and RECALLR_USER_ID must be set in the .env file")

    client = RecallrAI(api_key=api_key, project_id=project_id)

    # Get or create the user
    try:
        user = client.get_user(user_id)
        print(f"Found existing user: {user_id}")
    except UserNotFoundError:
        user = client.create_user(user_id=user_id)
        print(f"Created new user: {user_id}")

    # Fetch existing sessions to avoid dups based on original filename
    print("Fetching existing sessions to check for duplicates...")
    existing_filenames = set()
    offset = 0
    limit = 100
    while True:
        resp = user.list_sessions(offset=offset, limit=limit)
        for s in resp.sessions:
            if s.metadata and "filename" in s.metadata:
                existing_filenames.add(s.metadata["filename"])
        if len(resp.sessions) < limit:
            break
        offset += limit

    print(f"Found {len(existing_filenames)} previously ingested sessions")

    json_files = sorted(glob.glob(str(ingest_dir / "*.json")))
    if not json_files:
        print(f"No JSON files found in {ingest_dir}")
        return

    for jf in json_files:
        filename = Path(jf).name
        if filename in existing_filenames:
            print(f"Skipping {filename}: already ingested.")
            continue

        print(f"Ingesting {filename}...")
        with open(jf, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Failed to parse {filename}: {e}")
                continue

        # Fallback handling just in case the JSON is structured slightly differently
        timestamp_str = data["timestamp_utc"]
        # Replace Z with +00:00 to support older Python versions datetime.fromisoformat
        created_at = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        messages = data["messages"]

        # Create session with filename in metadata for future deduplication
        session = user.create_session(
            custom_created_at_utc=created_at,
            metadata={"filename": filename}
        )

        for msg in messages:
            # Add each message to the session
            role = MessageRole(msg["role"])
            content = msg["content"]
            session.add_message(role=role, content=content)

        # Trigger processing but do NOT wait
        session.process()
        print(f"Pushed {filename} to processing. Session ID: {session.session_id}")

    print("All ingestion tasks triggered!")


if __name__ == "__main__":
    main()
