import numpy as np
import pandas as pd
from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from transformers import TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from evaluate import load
import pickle
from datasets import Dataset
import re
import torch
import os
import matplotlib.pyplot as plt
from IPython.display import Audio
import pysrt
import librosa
import librosa.display
import seaborn as sns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)
pd.set_option("display.expand_frame_repr", False)

if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"Using device: {device}")

avspeech_train_path = '/Users/anchitmulye/Downloads/avspeech/avspeech_train.csv'
avspeech_test_path = '/Users/anchitmulye/Downloads/avspeech/avspeech_test.csv'
names = ["youtube_id", "start_segment", "end_segment", "x_coordinate", "y_coordinate"]

avspeech_train_df = pd.read_csv(avspeech_train_path, names=names)
avspeech_test_df = pd.read_csv(avspeech_test_path, names=names)

print(f"AVSpeech Train Dataset Shape: {avspeech_train_df.shape}")
print(f"AVSpeech Test Dataset Shape: {avspeech_test_df.shape}")

avspeech_train_df.head(2)

BASE_DIR = '/Users/anchitmulye/Downloads/train'
temp = []

for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)
    if os.path.isdir(folder_path):
        wav_path = ""
        text_path = ""
        for file in os.listdir(folder_path):
            if file.endswith("_masked_audio.wav"):
                wav_path = os.path.join(folder_path, file)
            if file.endswith(".srt"):
                text_path = os.path.join(folder_path, file)
        temp.append({
            "youtube_id": folder,
            "wav_path": wav_path,
            "text_path": text_path
        })

wav_df = pd.DataFrame(temp)
print(f"Total wav files:", wav_df.shape[0])

given_avspeech_df = pd.merge(avspeech_train_df, wav_df, on="youtube_id", how="inner")
print(f"Final Dataset Shape: {given_avspeech_df.shape}")

given_avspeech_df.head(3)

given_avspeech_df['duration'] = given_avspeech_df['end_segment'] - given_avspeech_df['start_segment']

plt.figure(figsize=(8, 5))
plt.hist(given_avspeech_df['duration'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Segment Durations")
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of Clips")
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 4))
sns.boxplot(x=given_avspeech_df['duration'], color='lightgreen')
plt.title("Boxplot of Audio Durations")
plt.xlabel("Duration (seconds)")
plt.grid(True)
plt.show()


sample_rows = given_avspeech_df.dropna(subset=['wav_path']).sample(3, random_state=42)

for _, row in sample_rows.iterrows():
    wav_path = row['wav_path']
    start_time = row['start_segment']
    end_time = row['end_segment']
    duration = end_time - start_time

    y, sr = librosa.load(wav_path, sr=None, offset=start_time, duration=duration)

    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform: {os.path.basename(wav_path)} [{start_time:.2f}s - {end_time:.2f}s]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


NUM_MFCC_COEFFICIENTS = 13
HOP_LENGTH = 512
FRAME_LENGTH = 2048
DURATION = 5


given_avspeech_df.head(1)


def extract_mfcc_from_segment(audio_path, start_segment, end_segment, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH,
                              n_mfcc=NUM_MFCC_COEFFICIENTS, duration=DURATION):
    if len(audio_path) == 0 or audio_path is None:
        return None

    duration = end_segment - start_segment
    y, sr = librosa.load(audio_path, offset=start_segment, duration=duration, sr=16000)

    if y.size == 0:
        print(f"Empty audio file found: {audio_path}")
        return None

    if len(y) < frame_length:
        pad_length = frame_length - len(y)
        y = np.pad(y, (0, pad_length), mode='constant')

    y = librosa.util.normalize(y)
    y = librosa.effects.preemphasis(y)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=frame_length, hop_length=hop_length)
    return y, sr, mfcc


def clean_caption_text(text):
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = text.replace('\\', '')
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    return text


def extract_text_from_segment(srt_path, start_time, end_time):
    if not os.path.exists(srt_path):
        return None
    try:
        subs = pysrt.open(srt_path)
        trimmed_lines = set()
        for sub in subs:
            sub_start = sub.start.ordinal / 1000
            sub_end = sub.end.ordinal / 1000
            if sub_end < start_time:
                continue
            if sub_start > end_time:
                break
            cleaned = clean_caption_text(sub.text.strip())
            if cleaned:
                trimmed_lines.add(cleaned)
        return ' '.join(sorted(trimmed_lines)) if trimmed_lines else None
    except Exception as e:
        print(f"Error processing {srt_path}: {e}")
        return None


processed_data = []
label_counter = {}
for index, row in given_avspeech_df.iterrows():
    start_segment = row['start_segment']
    end_segment = row['end_segment']
    wav_path = row['wav_path']
    text_path = row['text_path']

    mfcc_result = extract_mfcc_from_segment(wav_path, start_segment, end_segment)
    text_result = extract_text_from_segment(text_path, start_segment, end_segment)

    if mfcc_result is None:
        continue

    waveform, sampling_rate, mfcc = mfcc_result

    processed_data.append({
        "youtube_id": row['youtube_id'],
        "start_segment": start_segment,
        "end_segment": end_segment,
        "x_coordinate": row['x_coordinate'],
        "y_coordinate": row['y_coordinate'],
        "wav_path": wav_path,
        "text_path": row['text_path'],
        "trimmed_text": text_result,
        "duration": row['duration'],
        "waveform": waveform,
        "sampling_rate": sampling_rate,
        "mfcc": mfcc
    })

processed_df = pd.DataFrame(processed_data)
print(f"Features Dataframe Shape: {processed_df.shape}")

print("\nSample pre-processed data:")
print(f"Cleaned text: {processed_df.iloc[5]['trimmed_text']}")

y = processed_df.iloc[5]['waveform']
sr = processed_df.iloc[5]['sampling_rate']
Audio(y, rate=sr)

plt.figure(figsize=(15, 10))
index = 0
index_counter = 0

for _, row in processed_df.iterrows():

    if index_counter == 3:
        break

    mfcc = row['mfcc']

    num_frames = mfcc.shape[1]
    t = librosa.frames_to_time(range(num_frames), hop_length=HOP_LENGTH)

    plt.subplot(3, 1, index + 1)
    librosa.display.specshow(mfcc, x_coords=t, cmap='viridis', x_axis="time")
    plt.colorbar(format='%+2.0f dB')
    plt.ylabel("MFCC Coefficients")
    plt.title(f"Sample {index_counter + 1} — MFCC")

    index += 1
    index_counter += 1

plt.subplots_adjust(hspace=0.5)
plt.show()

# Load the model and processor
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_asr")
model = SpeechT5ForSpeechToText.from_pretrained("microsoft/speecht5_asr")


asr_df = processed_df[['waveform', 'trimmed_text']].rename(columns={'trimmed_text': 'text'})

asr_df['waveform'] = asr_df['waveform'].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
asr_df = asr_df[asr_df['text'].notna() & (asr_df['text'].str.strip() != '')].reset_index(drop=True)
asr_df.head(1)

# Split the dataset
train_df, test_df = train_test_split(asr_df, test_size=0.1, random_state=42)
train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)


def preprocess(example):
    audio_inputs = processor(
        audio=example["waveform"],
        sampling_rate=16000,
        return_tensors="pt"
    )
    text_labels = processor.tokenizer(
        example["text"],
        return_tensors="pt"
    )

    return {
        "input_values": audio_inputs.input_values[0],
        "labels": text_labels.input_ids[0]
    }


train_dataset = train_dataset.map(preprocess, remove_columns=["waveform", "text"])
test_dataset = test_dataset.map(preprocess, remove_columns=["waveform", "text"])

print("Train Dataset Size:", len(train_dataset))
print("Test Dataset Size:", len(test_dataset))

class CustomCTCCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_values = [torch.tensor(f["input_values"]) for f in features]
        labels = [torch.tensor(f["labels"]) for f in features]

        batch = {
            "input_values": pad_sequence(input_values, batch_first=True, padding_value=0.0),
            "labels": pad_sequence(labels, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id),
        }
        return batch

training_args = TrainingArguments(
    output_dir="/Users/anchitmulye/Downloads/output",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    logging_steps=20,
    save_steps=300,
    save_total_limit=2,
    fp16=False,
    bf16=False,
    no_cuda=True
)

data_collator = CustomCTCCollator(processor)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()

wer_metric = load("wer")


def compute_wer(model, processor, dataset):
    model.eval()
    predictions, references = [], []

    for sample in dataset:
        input_tensor = torch.tensor(sample["input_values"]).unsqueeze(0)
        with torch.no_grad():
            pred_ids = model.generate(input_tensor)
        pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].lower()

        label_text = processor.tokenizer.decode(sample["labels"], skip_special_tokens=True).lower()
        predictions.append(pred_text)
        references.append(label_text)

    return wer_metric.compute(predictions=predictions, references=references)


wer = compute_wer(model, processor, test_dataset)
print(f"WER on test set: {wer:.3f}")
