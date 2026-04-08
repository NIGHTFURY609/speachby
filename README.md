# 🎧 Speech Analysis System: Pause & Repetition Detection

## 📌 Overview
This project implements a Python-based system to analyze speech audio and detect:

- ⏸️ Pause segments (silent regions)
- 🔁 Repetition patterns (stuttering like "ba-ba-ball", "I-I-I want")

The system uses signal processing techniques (librosa) and acoustic feature analysis instead of relying on text-based detection.

---

## 🧠 Approach

The system follows a **three-stage pipeline**:

### 1. Audio Preprocessing
Audio is cleaned and normalized to ensure consistency across different recordings.

### 2. Feature Extraction
Important features like RMS energy and MFCCs are extracted to represent the speech signal.

### 3. Detection Logic
- Pauses are detected using energy thresholds
- Repetitions are detected using MFCC similarity

---

## 🎧 Audio Preprocessing

The audio is preprocessed using:

- **Normalization**
  - Scales audio between -1 and 1
  - Ensures consistent amplitude across inputs

- **Pre-emphasis filtering**
  - Enhances high-frequency components
  - Improves speech clarity

- **Silence trimming**
  - Removes unnecessary leading/trailing silence

---

## 📊 Feature Extraction

### 🔹 RMS Energy
- Measures signal strength over time
- Used for pause detection
- Low RMS → silence

### 🔹 MFCC (Mel-Frequency Cepstral Coefficients)
- Captures spectral characteristics of speech
- Mimics human auditory perception
- Used for repetition detection

---

## 🔍 Detection Logic

### ⏸️ Pause Detection

- RMS energy is computed frame-wise
- Adaptive threshold is applied
- Frames below threshold → silence
- Consecutive silent frames → pause segments

---

### 🔁 Repetition Detection (Audio-Based)

- MFCC features are extracted
- Audio is divided into small time windows
- Similarity between segments is computed using cosine distance

#### Logic:
- If two segments are highly similar → repetition
- Groups of similar segments → stutter pattern

#### Key condition:
cosine_distance < threshold


#### Additional constraints:
- Duration must be between 0.1s and 1.5s
- Avoids false positives from long speech segments

---

## 🗣️ Text Output

- Repetition is detected purely from audio
- Whisper is used **only to transcribe detected segments**
- No text-based detection is used

---

## ⚙️ Technologies Used

- Python
- librosa
- NumPy
- SciPy
- Whisper (for transcription only)

---

## 🚧 Challenges Faced

### 1. Low Audio Volume
- Caused poor feature extraction
- Solved using normalization

### 2. False Positives in Repetition
- Similar words triggered detection
- Solved using duration constraints

### 3. Over-grouping of Frames
- Entire audio was detected as repetition
- Fixed by limiting segment duration

### 4. Whisper Smoothing Stutters
- Whisper removes stutter patterns


---

## 🏆 Final Outcome

The system successfully:
- Detects pauses accurately
- Identifies repetition patterns from audio
- Outputs readable stutter patterns

---

## 📌 Future Improvements

- Use DTW (Dynamic Time Warping)
- Add real-time microphone support
- Improve phoneme-level detection
- Train ML model for stutter severity