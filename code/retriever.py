import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EXTRA_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'to', 'in', 'of', 'for', 'is', 'on', 'at', 'by', 'this', 'that', 'it', 
    'you', 'your', 'we', 'our', 'be', 'are', 'pls', 'please', 'hi', 'hello', 'dear', 'http', 'https', 'www', 
    'com', 'fwd', 'forwarded', 'image', 'ocr', 'voice', 'transcript', 'attachment'
}

def clean_text(text):
    """Clean text for tokenizing and TF-IDF processing."""
    if not text or not isinstance(text, str):
        return ""
    words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    filtered = [w for w in words if w not in EXTRA_STOPWORDS and len(w) > 1]
    return " ".join(filtered)

class EvidenceRetriever:
    def __init__(self, context_store):
        self.cs = context_store
        self.history_df = context_store.history_df
        self.events_df = context_store.events_df
        
        # Build event lookup map: message_id -> event dict
        self.event_map = {}
        if not self.events_df.empty:
            for _, row in self.events_df.iterrows():
                msg_id = str(row['message_id'])
                self.event_map[msg_id] = row.to_dict()

    def find_evidence(self, msg, msg_text, max_results=2, min_sim=0.12):
        """
        Find historical evidence message IDs for an incoming message.
        Uses TF-IDF text similarity, historical OCR/ASR, and user event boosting.
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

        cleaned_target = clean_text(msg_text)
        if not cleaned_target:
            return "none"

        # Prepare corpus of historical candidate texts (including full OCR/ASR)
        candidate_rows = []
        corpus = []
        for _, h_row in candidates.iterrows():
            h_dict = h_row.to_dict()
            h_full_text = self.cs.get_full_message_text(h_dict)
            cleaned_h = clean_text(h_full_text)
            if cleaned_h:
                candidate_rows.append(h_dict)
                corpus.append(cleaned_h)

        if not corpus:
            return "none"

        # Calculate TF-IDF Cosine Similarities
        try:
            vectorizer = TfidfVectorizer().fit(corpus + [cleaned_target])
            target_vec = vectorizer.transform([cleaned_target])
            corpus_vecs = vectorizer.transform(corpus)
            sim_matrix = cosine_similarity(target_vec, corpus_vecs)[0]
        except Exception:
            # Fallback if vectorizer encounters unexpected error
            sim_matrix = [0.0] * len(corpus)

        scored = []
        for idx, h_dict in enumerate(candidate_rows):
            h_id = str(h_dict['message_id'])
            sim = float(sim_matrix[idx])

            # Boost 1: Sender match
            if sender_user_id and str(h_dict.get('sender_user_id', '')) == sender_user_id:
                sim += 0.08

            # Boost 2: Media type match
            if str(h_dict.get('media_type', '')) == str(msg.get('media_type', '')) and pd.notna(msg.get('media_type')):
                sim += 0.05

            # Boost 3: Historical User Actions (Dismissed, Muted, Reported, Replied)
            event_info = self.event_map.get(h_id, {})
            if event_info:
                if int(event_info.get('muted_after_message', 0)) == 1 or int(event_info.get('message_reported', 0)) == 1:
                    sim += 0.15
                elif int(event_info.get('notification_dismissed', 0)) == 1:
                    sim += 0.10
                elif int(event_info.get('message_replied', 0)) == 1:
                    sim += 0.10

            if sim >= min_sim:
                scored.append((h_id, sim))

        if not scored:
            return "none"

        # Sort by similarity descending
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [item[0] for item in scored[:max_results]]
        
        return ";".join(top_ids)

