import os
import re
import pandas as pd
from collections import Counter

def tokenize(text):
    """Simple clean tokenizer for text similarity."""
    if not text or not isinstance(text, str):
        return set()
    words = re.findall(r'\w+', text.lower())
    # remove common stop words
    stopwords = {'the', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 'is', 'on', 'at', 'by', 'this', 'that', 'it', 'you', 'your', 'we', 'our', 'be', 'are', 'pls', 'please', 'hi', 'dear'}
    return {w for w in words if w not in stopwords and len(w) > 1}

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

class EvidenceRetriever:
    def __init__(self, context_store):
        self.cs = context_store
        self.history_df = context_store.history_df
        self.events_df = context_store.events_df

    def find_evidence(self, msg, msg_text, max_results=2, min_sim=0.18):
        """
        Find historical evidence message IDs for an incoming message.
        Returns a semicolon-separated string of message_ids, or 'none'.
        """
        if self.history_df.empty:
            return "none"

        user_id = str(msg.get('user_id', ''))
        conversation_type = str(msg.get('conversation_type', '')).strip().lower()
        group_id = str(msg.get('group_id', '')) if pd.notna(msg.get('group_id')) else ""
        business_id = str(msg.get('business_id', '')) if pd.notna(msg.get('business_id')) else ""
        sender_user_id = str(msg.get('sender_user_id', '')) if pd.notna(msg.get('sender_user_id')) else ""

        # Filter history for this user
        u_hist = self.history_df[self.history_df['user_id'] == user_id]
        if u_hist.empty:
            return "none"

        # Filter by entity match
        candidates = pd.DataFrame()
        if conversation_type == 'business' and business_id:
            candidates = u_hist[u_hist['business_id'] == business_id]
        elif conversation_type == 'group' and group_id:
            candidates = u_hist[u_hist['group_id'] == group_id]
            if sender_user_id and not candidates.empty:
                # prefer same sender if available
                sender_cand = candidates[candidates['sender_user_id'] == sender_user_id]
                if len(sender_cand) >= 1:
                    candidates = sender_cand
        elif conversation_type == 'personal' and sender_user_id:
            candidates = u_hist[u_hist['sender_user_id'] == sender_user_id]

        if candidates.empty:
            # Fallback to all user history if specific entity match returns empty
            candidates = u_hist

        target_tokens = tokenize(msg_text)
        if not target_tokens:
            return "none"

        scored = []
        for _, h_row in candidates.iterrows():
            h_id = str(h_row['message_id'])
            h_text = str(h_row.get('message_text', ''))
            h_tokens = tokenize(h_text)
            
            sim = jaccard_similarity(target_tokens, h_tokens)
            
            # Boost if sender or media matches
            if sender_user_id and str(h_row.get('sender_user_id', '')) == sender_user_id:
                sim += 0.05
            if str(h_row.get('media_type', '')) == str(msg.get('media_type', '')) and pd.notna(msg.get('media_type')):
                sim += 0.03

            if sim >= min_sim:
                scored.append((h_id, sim))

        if not scored:
            return "none"

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [item[0] for item in scored[:max_results]]
        
        return ";".join(top_ids)
