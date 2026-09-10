# video_backend.py
import cv2
import numpy as np
import tempfile
import os

# Dynamically locate the default frontal face Haar cascade built into OpenCV
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def analyze_video_liveness(video_bytes):
    """
    Analyzes video frame streams for facial boundary consistency and liveness cues.
    Returns:
        score (float): Confidence score between 0.0 and 1.0
        log_msg (str): Description of visual analysis verdict
    """
    if not video_bytes:
        return 0.50, "⚠️ No video payload received."

    # Write video bytes to a temporary file for OpenCV capture
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    try:
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            return 0.50, "⚠️ Unable to decode video stream."

        total_frames = 0
        detected_faces = 0
        frame_diffs = []
        prev_gray = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in frame
            if not face_detector.empty():
                faces = face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                if len(faces) > 0:
                    detected_faces += 1

            # Calculate frame-to-frame motion variance
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                frame_diffs.append(np.mean(diff))

            prev_gray = gray

        cap.release()

        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

        if total_frames == 0:
            return 0.50, "⚠️ Video stream contained 0 readable frames."

        # Calculate metrics
        face_detection_ratio = detected_faces / total_frames
        avg_motion = np.mean(frame_diffs) if frame_diffs else 0.0

        # Liveness Decision Logic
        if face_detection_ratio > 0.5:
            if avg_motion > 0.5:
                liveness_score = min(0.85 + (face_detection_ratio * 0.1), 0.98)
                log_msg = f"✅ High Visual Integrity: Face detected in {face_detection_ratio*100:.1f}% of frames with natural motion."
            else:
                liveness_score = 0.45
                log_msg = "⚠️ Static Frame Alert: High facial match but abnormally low motion variance (Potential Photo Replay)."
        else:
            liveness_score = 0.30
            log_msg = "🚨 Anomaly Detected: Low facial structural consistency across video stream."

        return liveness_score, log_msg

    except Exception as e:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        return 0.50, f"⚠️ Visual pipeline error: {str(e)}"