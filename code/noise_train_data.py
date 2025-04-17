import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pydub import AudioSegment

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

zero_lip_count = 0
delta_threshold = 15
lip_threshold = 10

mute_segments = []
current_mute_start = None
delta_seconds = 0.5  # Mute only if static for >= delta
delta_frames = int(30 * delta_seconds)
lip_static_counter = 0

# Load full CSV
df = pd.read_csv("/Users/anchitmulye/Downloads/avspeech/avspeech_train.csv",
                 header=None)  # filename, start_time, end_time

base_path = "/Users/anchitmulye/Downloads/train"

available_folders = set(os.listdir(base_path))

df = df[df[0].isin(available_folders)].reset_index(drop=True)
print(f"Filtered CSV to {len(df)} entries based on available folders in 'train'")


def trim_and_mute_audio(audio_path, start_time, end_time, mute_segments, output_path):
    audio = AudioSegment.from_file(audio_path)
    trimmed_audio = audio[start_time * 1000:end_time * 1000]
    for start_sec, end_sec in mute_segments:
        if start_sec is None or end_sec is None:
            print(f"Warning: Mute segment contains None values: {start_sec}, {end_sec} for {audio_path}")
            continue
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        silence = AudioSegment.silent(duration=end_ms - start_ms)
        trimmed_audio = trimmed_audio[:start_ms] + silence + trimmed_audio[end_ms:]
    try:
        trimmed_audio.export(output_path, format="wav")
    except Exception as e:
        print(f"File is having some issue {e} and skipped {audio_path}")


for idx, row in df.iterrows():
    name = row[0]
    start_time = row[1]  # in seconds
    end_time = row[2]

    print('\n')
    print(name, start_time, end_time)

    video_path = os.path.join(base_path, name, f"{name}.mp4")
    cap = cv2.VideoCapture(video_path)

    audio_path = os.path.join(base_path, name, f"{name}_audio_5_noise.wav")
    audio = AudioSegment.from_file(audio_path)
    audio_duration = len(audio) / 1000  # in seconds

    # Setup writer after loading properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    out_video_path = os.path.join(base_path, name, f"{name}_masked_output.mp4")
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    out_audio_path = os.path.join(base_path, name, f"{name}_masked_audio.wav")

    frame_count = start_frame

    while cap.isOpened() and frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        lip_distance = 0
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Light face mask overlay
                overlay = frame.copy()
                face_points = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    face_points.append([x, y])
                face_points = np.array([face_points], dtype=np.int32)
                cv2.fillPoly(overlay, face_points, color=(200, 230, 255))
                frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                # Draw face mesh
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

                # Lip distance logic
                top = face_landmarks.landmark[13]
                bottom = face_landmarks.landmark[14]
                top_coords = (int(top.x * w), int(top.y * h))
                bottom_coords = (int(bottom.x * w), int(bottom.y * h))
                lip_distance = abs(bottom_coords[1] - top_coords[1])
                cv2.line(frame, top_coords, bottom_coords, (0, 255, 0), 3)

                cv2.putText(frame, f"Lip Distance: {lip_distance:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # Smooth logic
        if lip_distance < lip_threshold:
            zero_lip_count += 1
        else:
            zero_lip_count = 0

        keep_audio = zero_lip_count < delta_threshold
        color = (0, 255, 0) if keep_audio else (0, 0, 255)
        cv2.putText(frame, f"Audio: {'ON' if keep_audio else 'OFF'}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if lip_distance < lip_threshold:
            lip_static_counter += 1
            if lip_static_counter == delta_frames:
                current_mute_start = (frame_count - start_frame) / fps - delta_seconds
                print(f"Lip inactivity start at: {current_mute_start}")
        else:
            if lip_static_counter >= delta_frames and current_mute_start is not None:
                current_time = (frame_count - start_frame) / fps
                print(f"Lip inactivity end at: {current_time}")
                mute_segments.append((current_mute_start, current_time))
            lip_static_counter = 0
            current_mute_start = None

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    # Edge case: mute at end
    if lip_static_counter >= delta_frames and current_mute_start is not None:
        mute_segments.append((current_mute_start, (end_frame - start_frame) / fps))

    # --- Save final muted audio
    trim_and_mute_audio(audio_path, start_time, end_time, mute_segments, out_audio_path)
    print("Muted audio exported.\n")

print("All videos processed and saved.")
