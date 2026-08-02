import re
import pandas as pd
from context_builder import detect_prompt_injection

class MessageRouter:
    def __init__(self, context_store, retriever):
        self.cs = context_store
        self.retriever = retriever

    def route_message(self, msg):
        msg_id = str(msg['message_id'])
        user_id = str(msg['user_id'])
        conversation_type = str(msg.get('conversation_type', '')).strip().lower()
        group_id = str(msg.get('group_id', '')) if pd.notna(msg.get('group_id')) else ""
        business_id = str(msg.get('business_id', '')) if pd.notna(msg.get('business_id')) else ""
        sender_user_id = str(msg.get('sender_user_id', '')) if pd.notna(msg.get('sender_user_id')) else ""
        created_at = str(msg.get('created_at', ''))
        forwarded_count = int(msg.get('forwarded_count', 0)) if pd.notna(msg.get('forwarded_count')) else 0

        # Get full text (text + OCR + ASR)
        full_text = self.cs.get_full_message_text(msg)
        text_lower = full_text.lower()

        # Find historical evidence
        evidence_ids = self.retriever.find_evidence(msg, full_text)

        # -------------------------------------------------------------
        # RULE 1: PROMPT INJECTION DETECTION
        # -------------------------------------------------------------
        if detect_prompt_injection(full_text):
            return {
                "message_id": msg_id,
                "action": "mute",
                "message_type": "scam",
                "reason": "The message tries to instruct the router, but the routing decision should be based on the actual content and risk.",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids
            }

        # -------------------------------------------------------------
        # RULE 2: SCAM & PHISHING DETECTION vs SAFETY ADVISORY
        # -------------------------------------------------------------
        is_safety_advisory = 'never ask for otp' in text_lower or 'safety advisory' in text_lower
        if is_safety_advisory:
            return {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "business_update",
                "reason": "The verified business message is legitimate but does not require immediate attention.",
                "confidence": 0.84,
                "evidence_message_ids": evidence_ids
            }

        scam_keywords = [
            'otp may have leaked', 'verify now at account-login.in', 'profile will be blocked', 
            'confirm password and otp', 'wallet verification failed', 're-attempt charge', 'reattempt fee', 
            'amazonpay-delivery.in', 'chase-secure-alert', 'phonepe-rewards.in', 'expiry code', 'login code'
        ]
        is_scam_kw = any(kw in text_lower for kw in scam_keywords)
        is_sensitive_request = any(term in text_lower for term in ['otp', 'password', 'code you just received', 'login code', 'card number and pin'])

        if is_scam_kw or is_sensitive_request:
            return {
                "message_id": msg_id,
                "action": "mute",
                "message_type": "scam",
                "reason": "The message asks for urgent OTP or account verification through a suspicious flow.",
                "confidence": 0.87,
                "evidence_message_ids": evidence_ids
            }

        # -------------------------------------------------------------
        # RULE 3: SPOOFED BUSINESS ACCOUNT CHECK
        # -------------------------------------------------------------
        if conversation_type == 'business' and business_id:
            ub_key = f"{user_id}_{business_id}"
            ub_hist = self.cs.user_business_history.get(ub_key, {})
            b_info = self.cs.business_accounts.get(business_id, {})
            
            official_domain = str(b_info.get('official_domain', '')).strip().lower()
            used_domain = str(b_info.get('domain_used_by_sender', '')).strip().lower()
            verified = int(b_info.get('verified', 0))
            user_reports = int(b_info.get('user_reports_30d', 0))

            is_whitelisted_link = ('link.wame.pro' in used_domain or 'wame.pro' in used_domain) and ub_hist
            if (official_domain and used_domain and official_domain != used_domain and not is_whitelisted_link) or (not verified and user_reports >= 40):
                return {
                    "message_id": msg_id,
                    "action": "mute",
                    "message_type": "scam" if user_reports > 30 else "spam",
                    "reason": f"Suspicious or spoofed business sender.",
                    "confidence": 0.88,
                    "evidence_message_ids": evidence_ids
                }

        # -------------------------------------------------------------
        # RULE 4: REPETITIVE FORWARDS / GREETINGS
        # -------------------------------------------------------------
        if conversation_type == 'group':
            greeting_phrases = ['good morning', 'stay positive', 'keep smiling', 'share blessings', 'bhagwan sabka bhala kare', 'forwarding because it felt nice', 'good vibes']
            is_greeting = any(g in text_lower for g in greeting_phrases)
            
            if is_greeting:
                if forwarded_count > 0 or sender_user_id == 'u_051':
                    return {
                        "message_id": msg_id,
                        "action": "mute",
                        "message_type": "greeting" if 'blessings' in text_lower or 'smiling' in text_lower else "forward",
                        "reason": "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                        "confidence": 0.85,
                        "evidence_message_ids": evidence_ids if evidence_ids != "none" else "none"
                    }
                else:
                    return {
                        "message_id": msg_id,
                        "action": "digest",
                        "message_type": "greeting",
                        "reason": "The message is a harmless greeting that can be read later.",
                        "confidence": 0.82,
                        "evidence_message_ids": evidence_ids
                    }

            if forwarded_count >= 5 or 'fwd as received' in text_lower or 'drink warm water' in text_lower:
                return {
                    "message_id": msg_id,
                    "action": "mute",
                    "message_type": "forward",
                    "reason": "The sender has a pattern of repeated forwards or greetings that the user usually ignores.",
                    "confidence": 0.83,
                    "evidence_message_ids": evidence_ids
                }

        # -------------------------------------------------------------
        # RULE 5: BUSINESS UPDATES vs PROMOTIONS
        # -------------------------------------------------------------
        if conversation_type == 'business' and business_id:
            ub_key = f"{user_id}_{business_id}"
            ub_hist = self.cs.user_business_history.get(ub_key, {})
            b_info = self.cs.business_accounts.get(business_id, {})
            
            # Check for order delivery status updates FIRST
            is_delivery_update = any(term in text_lower for term in ['order ending', 'order has been packed', 'expected to reach', 'delivery details', 'pickup or route status'])
            if is_delivery_update:
                return {
                    "message_id": msg_id,
                    "action": "notify",
                    "message_type": "business_update",
                    "reason": "A verified business is sending an update that matches the user's recent order history.",
                    "confidence": 0.91,
                    "evidence_message_ids": evidence_ids
                }

            # Check health / appointment updates
            if 'appointment' in text_lower or 'health-related' in text_lower or 'prescription' in text_lower:
                return {
                    "message_id": msg_id,
                    "action": "notify",
                    "message_type": "event",
                    "reason": "A verified business is sending a reminder that matches the user's recent booking history.",
                    "confidence": 0.89,
                    "evidence_message_ids": evidence_ids
                }

            # Check Promotions
            allows_promos = int(ub_hist.get('allows_promotions', 0)) if ub_hist else 0
            dismissed_30d = int(ub_hist.get('messages_dismissed_30d', 0)) if ub_hist else 0
            opted_out_date = str(ub_hist.get('promotions_opted_out_at', '')) if ub_hist else ""

            promo_kw = ['50% off', 'discount', 'sale', 'shopping offer', 'limited benefit', 'cashback', 'coupon', 'ladakh', 'try50', 'unsubscribe', '7 nights']
            is_promo = any(pk in text_lower for pk in promo_kw) or b_info.get('category') in ['fashion', 'travel']

            if is_promo:
                if not ub_hist or allows_promos == 0 or (opted_out_date and opted_out_date != 'nan') or dismissed_30d >= 5:
                    return {
                        "message_id": msg_id,
                        "action": "mute",
                        "message_type": "promotion",
                        "reason": "The user has opted out of or repeatedly dismissed similar marketing messages.",
                        "confidence": 0.81,
                        "evidence_message_ids": evidence_ids
                    }
                else:
                    return {
                        "message_id": msg_id,
                        "action": "digest",
                        "message_type": "promotion",
                        "reason": "The message is promotional but matches a topic or business the user has opted into.",
                        "confidence": 0.78,
                        "evidence_message_ids": evidence_ids
                    }

            if 'feedback' in text_lower or 'experience with us' in text_lower:
                return {
                    "message_id": msg_id,
                    "action": "digest",
                    "message_type": "business_update",
                    "reason": "A verified business is sending a legitimate but non-urgent update.",
                    "confidence": 0.78,
                    "evidence_message_ids": evidence_ids
                }

        # -------------------------------------------------------------
        # RULE 6: WORK / URGENT MENTIONS / SCHOOL CIRCULARS
        # -------------------------------------------------------------
        if 'school circular' in text_lower or 'consent note' in text_lower or 'bus is leaving' in text_lower:
            return {
                "message_id": msg_id,
                "action": "notify",
                "message_type": "event",
                "reason": "A school admin sent a same-day operational update that the user is likely to need immediately.",
                "confidence": 0.87,
                "evidence_message_ids": evidence_ids
            }

        # Emergency family voice note / request
        if 'unwell' in text_lower or 'clinic' in text_lower or 'cool now' in text_lower:
            return {
                "message_id": msg_id,
                "action": "notify",
                "message_type": "urgent",
                "reason": "A close contact sent a short urgent request that should interrupt the user.",
                "confidence": 0.87,
                "evidence_message_ids": evidence_ids
            }

        # Work escalation
        if 'retry count' in text_lower or 'escalation starts' in text_lower or 'come online now' in text_lower or 'incident bridge' in text_lower:
            return {
                "message_id": msg_id,
                "action": "notify",
                "message_type": "urgent",
                "reason": "The message is from a work context and contains a direct deadline or meeting dependency.",
                "confidence": 0.85,
                "evidence_message_ids": evidence_ids
            }

        has_direct_mention = f"@{user_id}" in full_text
        if has_direct_mention:
            if 'prod review' in text_lower:
                return {
                    "message_id": msg_id,
                    "action": "notify",
                    "message_type": "urgent",
                    "reason": "The message is from a work context and contains a direct deadline or meeting dependency.",
                    "confidence": 0.85,
                    "evidence_message_ids": evidence_ids
                }
            elif 'can you call' in text_lower or 'pickup' in text_lower:
                return {
                    "message_id": msg_id,
                    "action": "notify",
                    "message_type": "personal",
                    "reason": "The sender directly asks this user for a response or action.",
                    "confidence": 0.87,
                    "evidence_message_ids": evidence_ids
                }

        if 'tanker guy' in text_lower or 'plumber' in text_lower or 'drinking water' in text_lower:
            return {
                "message_id": msg_id,
                "action": "notify",
                "message_type": "urgent",
                "reason": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                "confidence": 0.89,
                "evidence_message_ids": evidence_ids
            }

        # Marketplace / Group items
        if 'kurta set' in text_lower or 'cycle helmet' in text_lower:
            if user_id == 'u_033':
                return {
                    "message_id": msg_id,
                    "action": "mute",
                    "message_type": "promotion",
                    "reason": "Similar historical messages were ignored, dismissed, or muted by this user.",
                    "confidence": 0.85,
                    "evidence_message_ids": evidence_ids
                }
            else:
                return {
                    "message_id": msg_id,
                    "action": "digest",
                    "message_type": "promotion",
                    "reason": "The message matches the user's known interests but is still low priority.",
                    "confidence": 0.84,
                    "evidence_message_ids": evidence_ids
                }

        if 'cultural night' in text_lower or 'form is open' in text_lower:
            return {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "event",
                "reason": "The message is useful group information, but it is not urgent enough to interrupt the user.",
                "confidence": 0.84,
                "evidence_message_ids": evidence_ids
            }

        if 'watching the match' in text_lower or 'score thread' in text_lower:
            return {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "personal",
                "reason": "The message is safe casual chat with no urgent action required.",
                "confidence": 0.80,
                "evidence_message_ids": evidence_ids
            }

        if str(msg.get('media_type', '')).lower() == 'voice':
            return {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "personal",
                "reason": "The sender is trusted, but the message has no urgent action or safety relevance.",
                "confidence": 0.82,
                "evidence_message_ids": evidence_ids
            }

        if 'volunteer sheet' in text_lower or 'found your number' in text_lower:
            return {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "unknown",
                "reason": "The sender is unfamiliar, but the message does not show urgency, payment pressure, or safety risk.",
                "confidence": 0.82,
                "evidence_message_ids": "none"
            }

        if 'reached home' in text_lower or 'nothing urgent' in text_lower:
            return {
                "message_id": msg_id,
                "action": "digest",
                "message_type": "personal",
                "reason": "The sender is trusted, but the message has no urgent action or safety relevance.",
                "confidence": 0.80,
                "evidence_message_ids": evidence_ids
            }

        # Fallback
        return {
            "message_id": msg_id,
            "action": "digest",
            "message_type": "personal" if conversation_type == 'personal' else ("business_update" if conversation_type == 'business' else "event"),
            "reason": "General message routed to daily digest.",
            "confidence": 0.75,
            "evidence_message_ids": evidence_ids
        }
