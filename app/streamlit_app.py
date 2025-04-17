import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

st.set_page_config(layout="wide")
st.title("💬 Lip Movement-Based Video Masking")

col1, col2 = st.columns(2)

with col1:
    video_file = st.file_uploader("🎥 Upload Input Video", type=["mp4", "avi", "mov"])
with col2:
    audio_file = st.file_uploader("🔊 Upload Input Audio (Optional)", type=["wav", "mp3"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
        tmp_video.write(video_file.read())
        tmp_video_path = tmp_video.name

    # output_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    # output_video_path = output_video_file.name
    output_video_path = f"{os.path.splitext(video_file.name)[0]}_masked.mp4"

    st.subheader("▶️ Input Video Preview")
    st.video(tmp_video_path)

    process = st.button("🚀 Process Video")

    if process:
        st.info("Processing video...")

        # ==== Setup MediaPipe ====
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(tmp_video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))

        threshold = 10
        zero_lip_count = 0
        delta_threshold = 15

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            lip_distance = 0
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape

                    overlay = frame.copy()
                    face_points = [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks.landmark]
                    face_points = np.array([face_points], dtype=np.int32)
                    cv2.fillPoly(overlay, face_points, color=(200, 230, 255))
                    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

                    top = face_landmarks.landmark[13]
                    bottom = face_landmarks.landmark[14]
                    top_coords = (int(top.x * w), int(top.y * h))
                    bottom_coords = (int(bottom.x * w), int(bottom.y * h))
                    lip_distance = abs(bottom_coords[1] - top_coords[1])
                    cv2.line(frame, top_coords, bottom_coords, (0, 255, 0), 3)

                    cv2.putText(frame, f"Lip Distance: {lip_distance:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            if lip_distance < threshold:
                zero_lip_count += 1
            else:
                zero_lip_count = 0

            keep_audio = zero_lip_count < delta_threshold
            color = (0, 255, 0) if keep_audio else (0, 0, 255)
            cv2.putText(frame, f"Audio: {'ON' if keep_audio else 'OFF'}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            out.write(frame)

        cap.release()
        out.release()
        face_mesh.close()

        st.success("✅ Processing Complete!")

        with open(output_video_path, "rb") as f:
            video_bytes = f.read()

        st.subheader("🎬 Output Masked Video")
        print(output_video_path)
        st.video(video_bytes)
