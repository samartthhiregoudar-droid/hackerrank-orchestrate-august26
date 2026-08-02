import os
import json
import re
import pandas as pd
from datetime import datetime, time

class ContextStore:
    def __init__(self, dataset_dir="dataset", media_cache_path=None):
        self.dataset_dir = dataset_dir
        if media_cache_path is None:
            media_cache_path = os.path.join(dataset_dir, "media_cache.json")
        
        self.media_cache = {"images": {}, "voice_notes": {}}
        if os.path.exists(media_cache_path):
            try:
                with open(media_cache_path, 'r', encoding='utf-8') as f:
                    self.media_cache = json.load(f)
            except Exception as e:
                print(f"Error loading media cache: {e}")

        # Load CSVs into dicts for fast O(1) lookup
        self.users = self._load_csv_dict("users.csv", key_col="user_id")
        self.groups = self._load_csv_dict("groups.csv", key_col="group_id")
        self.group_members = self._load_group_members()
        self.business_accounts = self._load_csv_dict("business_accounts.csv", key_col="business_id")
        self.user_business_history = self._load_user_business_history()
        
        # Load historical messages & events
        self.history_df = self._load_csv_df("message_history.csv")
        self.events_df = self._load_csv_df("message_events.csv")
        self.daily_summary = self._load_csv_dict("daily_notification_summary.csv", key_col="user_id")

    def _load_csv_df(self, filename):
        path = os.path.join(self.dataset_dir, filename)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def _load_csv_dict(self, filename, key_col):
        path = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        res = {}
        for _, row in df.iterrows():
            res[str(row[key_col])] = row.to_dict()
        return res

    def _load_group_members(self):
        path = os.path.join(self.dataset_dir, "group_members.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        res = {}
        for _, row in df.iterrows():
            key = f"{row['group_id']}_{row['user_id']}"
            res[key] = row.to_dict()
        return res

    def _load_user_business_history(self):
        path = os.path.join(self.dataset_dir, "user_business_history.csv")
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        res = {}
        for _, row in df.iterrows():
            key = f"{row['user_id']}_{row['business_id']}"
            res[key] = row.to_dict()
        return res

    def get_full_message_text(self, msg):
        """Combine text, image OCR, and voice note ASR."""
        parts = []
        msg_text = str(msg.get('message_text', '')).strip()
        if msg_text and msg_text.lower() != 'nan':
            parts.append(msg_text)

        media_type = str(msg.get('media_type', '')).strip().lower()
        media_id = str(msg.get('media_id', '')).strip()

        if media_type == 'image' and media_id in self.media_cache.get('images', {}):
            ocr_text = self.media_cache['images'][media_id]
            if ocr_text:
                parts.append(f"[IMAGE OCR]: {ocr_text}")
        elif media_type == 'voice' and media_id in self.media_cache.get('voice_notes', {}):
            asr_text = self.media_cache['voice_notes'][media_id]
            if asr_text:
                parts.append(f"[VOICE TRANSCRIPT]: {asr_text}")

        return "\n".join(parts)

    def is_business_spoofed(self, business_id):
        """Check if business domain used by sender matches official domain or is reported."""
        b_info = self.business_accounts.get(str(business_id), {})
        if not b_info:
            return False, "Unknown business"
        
        official = str(b_info.get('official_domain', '')).strip().lower()
        used = str(b_info.get('domain_used_by_sender', '')).strip().lower()
        verified = int(b_info.get('verified', 0))
        reports = int(b_info.get('user_reports_30d', 0))
        
        if official and used and official != used:
            return True, f"Domain mismatch (Official: {official}, Used: {used})"
        if not verified and reports >= 40:
            return True, f"Unverified account with high user reports ({reports})"
        
        return False, "Valid business"

    def is_dnd_time(self, user_id, timestamp_str):
        """Check if timestamp falls within user's DND window."""
        u_info = self.users.get(str(user_id), {})
        dnd_win = str(u_info.get('do_not_disturb_window', '')).strip()
        if not dnd_win or '-' not in dnd_win:
            return False
        
        try:
            start_str, end_str = dnd_win.split('-')
            start_t = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_t = datetime.strptime(end_str.strip(), "%H:%M").time()
            
            dt = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M")
            msg_t = dt.time()
            
            if start_t > end_t: # Overnight window (e.g. 22:00 - 07:00)
                return msg_t >= start_t or msg_t <= end_t
            else:
                return start_t <= msg_t <= end_t
        except Exception:
            return False

def detect_prompt_injection(text):
    """Detect attempts to override router instructions."""
    patterns = [
        r"ignore all previous",
        r"routing override",
        r"set action=",
        r"mark this message as",
        r"instruct the router",
        r"system prompt",
        r"override rules"
    ]
    text_lower = text.lower()
    for p in patterns:
        if re.search(p, text_lower):
            return True
    return False
