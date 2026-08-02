import os
import json
import subprocess
import pandas as pd

def detect_visual_photo_with_opencv(img_path):
    """Use OpenCV to check if image is a visual photo (e.g., product/clothing) when OCR is empty."""
    try:
        import cv2
        img = cv2.imread(img_path)
        if img is not None:
            # High color standard deviation indicates a real photo (clothes, products, scene)
            if img.std() > 25.0:
                return "[ATTACHMENT: Visual Product Photo / Image]"
    except Exception:
        pass
    return ""

def extract_ocr_text(img_path):
    """
    Extract text from image using PaddleOCR / Tesseract OCR CLI.
    If no text is found, uses OpenCV to detect visual photo attachments.
    """
    if not os.path.exists(img_path):
        return ""
    
    text = ""

    # 1. Try PaddleOCR if available
    try:
        from paddleocr import PaddleOCR
        ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        result = ocr_engine.ocr(img_path, cls=True)
        if result and result[0]:
            lines = [line[1][0] for line in result[0] if line and line[1]]
            text = " ".join(lines).strip()
            if text:
                return text
    except (ImportError, Exception):
        pass

    # 2. Fall back to Tesseract OCR CLI (v5.5)
    try:
        res = subprocess.run(['tesseract', img_path, 'stdout'], capture_output=True, timeout=15)
        text = res.stdout.decode('utf-8', errors='ignore').strip()
    except Exception as e:
        print(f"OCR Error processing {img_path}: {e}")

    # 3. If image has no OCR text (like img_008 clothing photo), use OpenCV to detect visual photo attachment
    if not text:
        visual_tag = detect_visual_photo_with_opencv(img_path)
        if visual_tag:
            return visual_tag

    return text

def extract_asr_text(audio_path, whisper_model=None):
    """
    Extract transcript from audio using faster-whisper (if installed) or openai-whisper as default.
    """
    if not os.path.exists(audio_path):
        return ""

    # 1. Try faster-whisper (CTranslate2 backend)
    try:
        from faster_whisper import WhisperModel
        fw_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        segments, _ = fw_model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments]).strip()
        if text:
            return text
    except (ImportError, Exception):
        pass

    # 2. Fall back to OpenAI Whisper
    try:
        if whisper_model is None:
            import whisper
            whisper_model = whisper.load_model('tiny')
        res = whisper_model.transcribe(audio_path)
        return res.get('text', '').strip()
    except Exception as e:
        print(f"ASR Error processing {audio_path}: {e}")
        return ""

def load_or_build_media_cache(dataset_dir="dataset", cache_file=None):
    """Load existing media cache or generate OCR/ASR for all media files."""
    if cache_file is None:
        cache_file = os.path.join(dataset_dir, "media_cache.json")
    
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                print(f"Loading cached media features from {cache_file}")
                return json.load(f)
        except Exception as e:
            print(f"Failed to read cache file {cache_file}, rebuilding: {e}")

    print("Extracting OCR and ASR features from media files...")
    cache = {"images": {}, "voice_notes": {}}

    # Process Images
    images_csv = os.path.join(dataset_dir, "images.csv")
    if os.path.exists(images_csv):
        img_df = pd.read_csv(images_csv)
        for _, row in img_df.iterrows():
            img_id = str(row['image_id'])
            rel_path = str(row['file_path'])
            full_path = os.path.join(dataset_dir, rel_path)
            ocr_text = extract_ocr_text(full_path)
            cache["images"][img_id] = ocr_text
            print(f"OCR [{img_id}]: {ocr_text[:60]}...")

    # Process Voice Notes
    vn_csv = os.path.join(dataset_dir, "voice_notes.csv")
    if os.path.exists(vn_csv):
        import whisper
        w_model = whisper.load_model('tiny')
        vn_df = pd.read_csv(vn_csv)
        for _, row in vn_df.iterrows():
            vn_id = str(row['voice_note_id'])
            rel_path = str(row['file_path'])
            full_path = os.path.join(dataset_dir, rel_path)
            asr_text = extract_asr_text(full_path, whisper_model=w_model)
            cache["voice_notes"][vn_id] = asr_text
            print(f"ASR [{vn_id}]: {asr_text[:60]}...")

    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
        print(f"Media cache saved to {cache_file}")
    except Exception as e:
        print(f"Failed to save cache file: {e}")

    return cache

if __name__ == "__main__":
    cache = load_or_build_media_cache()
    print(f"Processed {len(cache['images'])} images and {len(cache['voice_notes'])} voice notes.")
