import librosa
import numpy as np
from scipy.spatial.distance import cosine
import whisper
import tempfile
import soundfile as sf
import warnings
import sys
import io

# Fix Unicode encoding issues on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Suppress warnings from Whisper/Librosa for cleaner output
warnings.filterwarnings("ignore")

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Model loaded!\n")

# -----------------------------
# CONFIG
# -----------------------------
CONFIG = {
    "frame_length": 2048,
    "hop_length": 512,
    "mfcc_n": 13,
    "threshold_method": "percentile",
    "pause_percentile": 10,
    "similarity_thresh": 0.25,  
    "min_repetition_frames": 3
}

# -----------------------------
# 1. LOAD AUDIO
# -----------------------------
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y)) 
    y = librosa.effects.preemphasis(y)
    y, _ = librosa.effects.trim(y, top_db=20)
    return y, sr

# -----------------------------
# 2. FEATURE EXTRACTION
# -----------------------------
def extract_rms(y, cfg):
    rms = librosa.feature.rms(y=y, frame_length=cfg["frame_length"], hop_length=cfg["hop_length"])[0]
    rms = np.convolve(rms, np.ones(5)/5, mode='same')
    return rms

def extract_mfcc(y, sr, cfg):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=cfg["mfcc_n"], hop_length=cfg["hop_length"], n_fft=cfg["frame_length"])
    mfcc = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3)/3, mode='same'), axis=1, arr=mfcc)
    return mfcc

# -----------------------------
# 3. THRESHOLD & PAUSES
# -----------------------------
def get_adaptive_threshold(rms, cfg):
    return np.percentile(rms, 25) * 0.8

def detect_pauses(y, sr, rms, threshold, cfg):
    pauses = []
    is_pause = False
    start = 0
    for i, energy in enumerate(rms):
        time = i * cfg["hop_length"] / sr
        if energy < threshold:
            if not is_pause:
                is_pause = True
                start = time
        else:
            if is_pause:
                pauses.append((start, time))
                is_pause = False
    if is_pause:
        pauses.append((start, len(y)/sr))
    total_pause = sum(e - s for s, e in pauses)
    return pauses, total_pause

# -----------------------------
# 4. AUDIO REPETITION (MFCC ONLY)
# -----------------------------
def detect_repetition_patterns(mfcc, sr, cfg):
    events = []
    # Skip ahead ~200ms to compare separate syllables, not overlapping frames
    window = 6 
    i = 0

    while i < mfcc.shape[1] - window:
        # Skip absolute silence frames to avoid false positives on background noise
        if np.max(np.abs(mfcc[:, i])) == 0:
            i += 1
            continue

        group = [i]
        j = i + window
        
        while j < mfcc.shape[1]:
            dist = cosine(mfcc[:, i], mfcc[:, j])
            if dist < cfg["similarity_thresh"]:
                group.append(j)
                j += window
            else:
                break

        # Calculate physical duration of the repetition in seconds
        duration = (group[-1] - group[0]) * cfg["hop_length"] / sr
        
        # A valid stutter is usually between 0.1s and 1.5s. 
        if 0.1 < duration < 1.5 and len(group) >= 2: 
            events.append(group)
            i = group[-1] # Skip past this repetition so we don't count it twice
        else:
            i += 1

    return events

# -----------------------------
# 5. AUDIO SEGMENT & TEXT FORMATTERS
# -----------------------------
def extract_audio_segment(y, sr, start, end):
    return y[int(start * sr): int(end * sr)]

def transcribe_segment(audio_segment, sr):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio_segment, sr)
        # Force Whisper to see stutters for this specific audio slice
        result = whisper_model.transcribe(tmp.name, initial_prompt="b-b-ba")
    return result["text"].strip()

def format_stutter_text(text, repeat_count):
    words = text.lower().split()
    if not words: return ""
    if len(words) == 1:
        base = words[0]
        return "-".join([base[:2]] * repeat_count + [base])
    first = words[0]
    if all(w == first for w in words[:-1]):
        return "-".join([first] * repeat_count + [words[-1]])
    if words[-1].startswith(first):
        return "-".join([first] * repeat_count + [words[-1]])
    return text

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def analyze_audio(file_path, cfg=CONFIG):
    y, sr = preprocess_audio(file_path)
    duration = len(y) / sr

    rms = extract_rms(y, cfg)
    mfcc = extract_mfcc(y, sr, cfg)
    threshold = get_adaptive_threshold(rms, cfg)

    # Calculate Pauses & Repetitions (Audio Only)
    pauses, total_pause = detect_pauses(y, sr, rms, threshold, cfg)
    repetition_groups = detect_repetition_patterns(mfcc, sr, cfg)

    # -----------------------------
    # OUTPUT COMPILATION
    # -----------------------------
    print(f"\n--- Analysis Results for: {file_path} ---")

    print("\n[Pause Segments]")
    formatted_pauses = [f"[{s:.1f}s - {e:.1f}s]" for s, e in pauses]
    print(", ".join(formatted_pauses) if formatted_pauses else "No pauses detected.")
    print(f"Total Pause Duration: {total_pause:.1f}s")

    stuttered_words = []

    # Process AUDIO-BASED repetitions into text
    if repetition_groups:
        for group in repetition_groups:
            # We add a tiny buffer (0.1s before, 0.2s after) to give Whisper enough context to catch the word
            start = max(0, (group[0] * cfg["hop_length"] / sr) - 0.1)
            end = min(duration, (group[-1] * cfg["hop_length"] / sr) + 0.2)

            segment_audio = extract_audio_segment(y, sr, start, end)
            text = transcribe_segment(segment_audio, sr)

            # Only format if it's a short phrase (prevents formatting whole sentences if a false positive occurs)
            if text and len(text.split()) <= 3:
                formatted_text = format_stutter_text(text, len(group)-1)
                stuttered_words.append(formatted_text)

    # Clean up array: Remove duplicates and stray punctuation from Whisper
    stuttered_words = list(dict.fromkeys(stuttered_words))
    stuttered_words = [word.replace(".", "").replace(",", "").replace("?", "") for word in stuttered_words]

    # Print the exact UI Format
    print("\n🔁 Stuttered Words")
    if stuttered_words:
        print(", ".join(stuttered_words))
    else:
        print("None detected")

    print(f"\nCount: {len(stuttered_words)}\n")

    return pauses, stuttered_words

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    test_file = "./speech-stu-2.wav" 
    try:
        analyze_audio(test_file)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{test_file}'. Please check the path.")