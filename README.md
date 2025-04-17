# speech-understanding-project
### Akansha Gautam (M23CSA506)
### Anchit Mulye (M23CSA507)

---

## Methodology

1. **Dataset**: We used the AVSpeech dataset and selected 100 English-speaking YouTube videos using Whisper-tiny. For each sample, we extracted:
   - Audio (original)
   - Video (with visible face)
   - Auto-generated subtitles

2. **Audio Variants**:
   - **Original**: Clean audio from AVSpeech
   - **Noisy**: Added background noise using NVIDIA Riva
   - **Cleaned**: Applied lip-reading-based noise suppression (used lip-distance from MediaPipe to mute non-speech segments)

3. **Models**:
   - Fine-tuned three separate SpeechT5 models on each audio type using corresponding transcripts.

---

## 📊 Results

We evaluated all three models using **Word Error Rate (WER)**:

| Model                        | WER    |
|-----------------------------|--------|
| Original Audio              | 1.067  |
| Noisy Audio                 | 1.123  |
| Cleaned Audio (Lip-Reading) | 1.082  |

The model trained on cleaned audio (with lip-reading) performed much better than the one trained on noisy data, and came close to the clean baseline.

---

## 💡 What We Learned

- Adding noise degrades ASR performance significantly.
- Lip-reading helps recover accuracy by muting irrelevant segments.
- Visual information is very useful for robust speech recognition.

---
