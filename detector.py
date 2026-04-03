import cv2
import numpy as np
from scipy.spatial import distance as dist
import time
import pygame
import threading
import pyttsx3


# ── Initialize alarm
pygame.mixer.init()

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def play_alarm(message="Wake up! You are drowsy!"):
    def _speak():
        engine.say(message)
        engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# ── Thresholds
EAR_CONSEC_FRAMES  = 15
YAWN_CONSEC_FRAMES = 20

# ── Load OpenCV built-in models
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def run_detector():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    eye_frames   = 0
    yawn_frames  = 0
    total_blinks = 0
    total_yawns  = 0
    is_drowsy    = False
    is_yawning   = False
    start_time   = time.time()

    print("[INFO] Detector started — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        alert_level  = "SAFE"
        eyes_visible = 0
        yawn_detected = False

        for (fx, fy, fw, fh) in faces[:1]:
            # Draw face box
            cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 200), 1)

            roi_gray  = gray[fy:fy+fh, fx:fx+fw]
            roi_color = frame[fy:fy+fh, fx:fx+fw]

            # ── Eye detection (upper half of face)
            upper_gray = roi_gray[0:fh//2, :]
            eyes = eye_cascade.detectMultiScale(upper_gray, 1.1, 10, minSize=(20, 20))
            eyes_visible = len(eyes)

            for (ex, ey, ew, eh) in eyes:
                cx = fx + ex + ew // 2
                cy = fy + ey + eh // 2
                cv2.circle(frame, (cx, cy), 3, (0, 255, 200), -1)
                cv2.rectangle(frame,
                    (fx+ex, fy+ey),
                    (fx+ex+ew, fy+ey+eh),
                    (0, 255, 200), 1)

            # ── Mouth/yawn detection (lower half of face)
            lower_gray = roi_gray[fh//2:, :]
            mouths = mouth_cascade.detectMultiScale(lower_gray, 1.5, 20, minSize=(30, 20))
            yawn_detected = len(mouths) > 0

            if yawn_detected:
                for (mx, my, mw, mh) in mouths[:1]:
                    cv2.rectangle(frame,
                        (fx+mx, fy+fh//2+my),
                        (fx+mx+mw, fy+fh//2+my+mh),
                        (0, 200, 255), 1)

        # ── Drowsiness logic
        if len(faces) > 0:
            if eyes_visible == 0:
                eye_frames += 1
            else:
                if eye_frames >= 3:
                    total_blinks += 1
                eye_frames = 0
                is_drowsy  = False

            if eye_frames >= EAR_CONSEC_FRAMES:
                is_drowsy   = True
                alert_level = "DANGER"
                if time.time() - last_alarm_time > alarm_cooldown:
                   last_alarm_time = time.time()
                   threading.Thread(target=play_alarm, daemon=True).start()
            # ── Yawn logic
            if yawn_detected:
                yawn_frames += 1
            else:
                if yawn_frames >= YAWN_CONSEC_FRAMES:
                    total_yawns += 1
                yawn_frames = 0
                is_yawning  = False
                last_alarm_time  = 0
                alarm_cooldown   = 5  # seconds between alarms

            if yawn_frames >= YAWN_CONSEC_FRAMES:
                is_yawning = True
                if alert_level != "DANGER":
                    alert_level = "WARNING"
                    if time.time() - last_alarm_time > alarm_cooldown:
                       last_alarm_time = time.time()
                       threading.Thread(target=lambda: play_alarm("Yawning detected! Take a break!"), daemon=True).start()



        # ── Session time
        elapsed = int(time.time() - start_time)
        mins, secs = divmod(elapsed, 60)

        # ── Status bar top
        color_map = {
            "SAFE"   : (0, 220, 120),
            "WARNING": (0, 200, 255),
            "DANGER" : (0, 50,  255)
        }
        col = color_map[alert_level]

        cv2.rectangle(frame, (0, 0), (w, 40), (15, 15, 25), -1)
        cv2.putText(frame, f"STATUS: {alert_level}", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w-80, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

        # ── Bottom panel
        cv2.rectangle(frame, (0, h-50), (w, h), (15, 15, 25), -1)
        cv2.putText(frame, f"Eyes: {eyes_visible}", (10, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
        cv2.putText(frame, f"Blinks: {total_blinks}", (120, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
        cv2.putText(frame, f"Yawns: {total_yawns}", (260, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
        cv2.putText(frame, f"Face: {'YES' if len(faces)>0 else 'NO'}", (380, h-28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

        # ── Big alert messages
        if alert_level == "DANGER":
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.putText(frame, "! WAKE UP !", (w//2-130, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 50, 255), 3)

        if alert_level == "WARNING":
            cv2.putText(frame, "YAWNING DETECTED", (w//2-150, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)

        cv2.imshow("DriveSafe AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector()