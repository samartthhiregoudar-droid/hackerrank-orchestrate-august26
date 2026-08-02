import os
import pandas as pd
from media_processor import load_or_build_media_cache
from context_builder import ContextStore
from retriever import EvidenceRetriever
from router import MessageRouter

def main():
    print("=" * 60)
    print("HackerRank Orchestrate: WhatsApp Message Notification Router")
    print("=" * 60)

    dataset_dir = "dataset"
    messages_csv = os.path.join(dataset_dir, "messages.csv")
    output_csv = os.path.join(dataset_dir, "output.csv")

    if not os.path.exists(messages_csv):
        print(f"Error: {messages_csv} not found!")
        return

    # 1. Load / process media OCR and ASR
    media_cache = load_or_build_media_cache(dataset_dir)

    # 2. Build Context Store
    print("Building relational context and indexing user history...")
    context_store = ContextStore(dataset_dir)

    # 3. Initialize Retriever and Router
    retriever = EvidenceRetriever(context_store)
    router = MessageRouter(context_store, retriever)

    # 4. Read incoming messages
    messages_df = pd.read_csv(messages_csv)
    print(f"Loaded {len(messages_df)} incoming messages from {messages_csv}.")

    output_rows = []

    # 5. Process each message
    for idx, row in messages_df.iterrows():
        msg_dict = row.to_dict()
        decision = router.route_message(msg_dict)
        output_rows.append(decision)

    # 6. Build Output DataFrame
    output_df = pd.DataFrame(output_rows)
    
    # Ensure exact column order required by contract
    required_cols = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]
    output_df = output_df[required_cols]

    # Save to dataset/output.csv
    output_df.to_csv(output_csv, index=False)
    print(f"Successfully generated predictions for {len(output_df)} messages -> {output_csv}")

    # Action Summary
    action_counts = output_df['action'].value_counts().to_dict()
    print("\nRouting Summary by Action:")
    for act, count in action_counts.items():
        print(f"  {act}: {count}")

    type_counts = output_df['message_type'].value_counts().to_dict()
    print("\nRouting Summary by Message Type:")
    for mtype, count in type_counts.items():
        print(f"  {mtype}: {count}")

    print("=" * 60)

if __name__ == "__main__":
    main()
