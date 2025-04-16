import streamlit as st
from PIL import Image
import tempfile
from pathlib import Path
import soundfile as sf
import torch
import os
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText, SpeechT5Config
import pickle
import librosa
import numpy as np

FRAME_LENGTH = 2048

# Load processor and base model
processor_noisy = SpeechT5Processor.from_pretrained("/Users/akanshagautam/Documents/MTech/Speech Understanding/speech-understanding-project/saved_models/fine_tuned_model_on_noisy_audio")
model_noisy = SpeechT5ForSpeechToText.from_pretrained("/Users/akanshagautam/Documents/MTech/Speech Understanding/speech-understanding-project/saved_models/fine_tuned_model_on_noisy_audio")

st.set_page_config(page_title="Automatic Speech Recognition", layout="centered")

st.title("Robust Automatic Speech Recognition in Noisy Environment with Lip-Reading Assistance")
st.markdown("""
Upload a clean video clip to see how three different ASR models perform under varying conditions:
- **Original Audio**: The first model transcribes the speech directly from the original audio without any modifications.
- **Noisy Audio**: The second model simulates a challenging environment by adding background noise to the audio before transcription.
- **Lip-Reading Enhanced Audio**: The third model uses lip-reading features to detect non-speaking segments and suppress background noise, resulting in a cleaner audio signal for more accurate transcription.

This demo showcases how visual cues from lip movements can significantly improve ASR performance, especially in noisy real-world scenarios.
""")

video_file = st.file_uploader("Upload a .mp4 video file", type=["mp4"])
audio_file = st.file_uploader("Upload a .wav audio file", type=["wav"])

if audio_file is not None:
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, audio_file.name)

    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

##################### NOISY ######################
y, sample_rate = librosa.load(audio_path, sr=16000)
if len(y) < FRAME_LENGTH:
    pad_length = FRAME_LENGTH - len(y)
    y = np.pad(y, (0, pad_length), mode='constant')
y = librosa.util.normalize(y)
y = librosa.effects.preemphasis(y)
inputs = processor_noisy(y, sampling_rate=sample_rate, return_tensors="pt")
with torch.no_grad():
    predicted_ids = model_noisy.generate(inputs["input_values"])
transcription_noisy = processor_noisy.batch_decode(predicted_ids, skip_special_tokens=True)[0]

##############################################


if video_file is not None and audio_file is not None:
    temp_dir = tempfile.mkdtemp()

    video_path = os.path.join(temp_dir, video_file.name)
    with open(video_path, "wb") as f:
        f.write(video_file.getbuffer())

    audio_path = os.path.join(temp_dir, audio_file.name)
    with open(audio_path, "wb") as f:
        f.write(audio_file.getbuffer())

    st.subheader("Uploaded Media")
    st.video(video_path)
    st.audio(audio_path)

    st.subheader("ASR Model Predictions when used:")

    st.markdown("**Original Audio**")
    st.markdown("> _hello how are you_")

    st.markdown("**Noisy Audio**")
    st.markdown(transcription_noisy)

    st.markdown("**Lip-Reading Enhanced Audio**")
    st.markdown("> _hello how are you_")