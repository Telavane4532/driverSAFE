import cv2
import numpy as np
import time
import threading
import pyttsx3

# ── Thresholds
EAR_CONSEC_FRAMES  = 15
YAWN_CONSEC_FRAMES = 20
ALARM_COOLDOWN     = 5

# ── Cascades
face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

# ── Voice
def speak(msg):
    def _s():
        eng = pyttsx3.init()
        eng.setProperty('rate', 150)
        eng.setProperty('volume', 1.0)
        eng.say(msg)
        eng.runAndWait()
        eng.stop()
    threading.Thread(target=_s, daemon=True).start()

class DrowsinessCore:
    def __init__(self):
        self._frame       = None
        self._lock        = threading.Lock()
        self.eye_frames   = 0
        self.yawn_frames  = 0
        self.total_blinks = 0
        self.total_yawns  = 0
        self.is_drowsy    = False
        self.is_yawning   = False
        self.alert_level  = "SAFE"
        self.risk_score   = 0
        self.last_alarm   = 0
        self.start_time   = time.time()
        self.alert_log    = []
        self.fps          = 0
        self.face_detected = False
        self.eyes_visible  = 0

    def get_frame(self):
        return self._frame

    def get_stats(self):
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        return {
            "alert_level"  : self.alert_level,
            "risk_score"   : self.risk_score,
            "total_blinks" : self.total_blinks,
            "total_yawns"  : self.total_yawns,
            "is_drowsy"    : self.is_drowsy,
            "is_yawning"   : self.is_yawning,
            "face_detected": self.face_detected,
            "eyes_visible" : self.eyes_visible,
            "session_time" : f"{mins:02d}:{secs:02d}",
            "fps"          : round(self.fps, 1),
            "alert_log"    : self.alert_log[-8:],
        }

    def _log_alert(self, msg):
        self.alert_log.append({
            "time": time.strftime("%H:%M:%S"),
            "msg" : msg
        })
        if len(self.alert_log) > 50:
            self.alert_log.pop(0)

    def _alarm(self, msg):
        if time.time() - self.last_alarm > ALARM_COOLDOWN:
            self.last_alarm = time.time()
            self._log_alert(msg)
            speak(msg)

    def _calc_risk(self):
        score = 0
        score += min(self.eye_frames * 4, 60)
        score += min(self.yawn_frames * 2, 30)
        score += min(self.total_yawns * 2, 10)
        self.risk_score = min(int(score), 100)

    def start(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        fps_counter = 0
        fps_start   = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

            self.alert_level  = "SAFE"
            self.eyes_visible = 0
            yawn_detected     = False
            self.face_detected = len(faces) > 0

            for (fx, fy, fw, fh) in faces[:1]:
                cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), (0, 255, 200), 1)

                roi_gray = gray[fy:fy+fh, fx:fx+fw]

                # Eyes
                upper = roi_gray[0:fh//2, :]
                eyes  = eye_cascade.detectMultiScale(upper, 1.1, 10, minSize=(20, 20))
                self.eyes_visible = len(eyes)

                for (ex, ey, ew, eh) in eyes:
                    cx = fx + ex + ew // 2
                    cy = fy + ey + eh // 2
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 200), -1)
                    cv2.rectangle(frame,
                        (fx+ex, fy+ey),
                        (fx+ex+ew, fy+ey+eh),
                        (0, 255, 200), 1)

                # Mouth
                lower  = roi_gray[fh//2:, :]
                mouths = mouth_cascade.detectMultiScale(lower, 1.5, 20, minSize=(30, 20))
                yawn_detected = len(mouths) > 0

                if yawn_detected:
                    for (mx, my, mw, mh) in mouths[:1]:
                        cv2.rectangle(frame,
                            (fx+mx, fy+fh//2+my),
                            (fx+mx+mw, fy+fh//2+my+mh),
                            (0, 200, 255), 1)

            # ── Drowsiness
            if self.face_detected:
                if self.eyes_visible == 0:
                    self.eye_frames += 1
                else:
                    if self.eye_frames >= 3:
                        self.total_blinks += 1
                    self.eye_frames = 0
                    self.is_drowsy  = False

                if self.eye_frames >= EAR_CONSEC_FRAMES:
                    self.is_drowsy   = True
                    self.alert_level = "DANGER"
                    self._alarm("Wake up! You are drowsy!")

                # ── Yawn
                if yawn_detected:
                    self.yawn_frames += 1
                else:
                    if self.yawn_frames >= YAWN_CONSEC_FRAMES:
                        self.total_yawns += 1
                    self.yawn_frames = 0
                    self.is_yawning  = False

                if self.yawn_frames >= YAWN_CONSEC_FRAMES:
                    self.is_yawning = True
                    if self.alert_level != "DANGER":
                        self.alert_level = "WARNING"
                        self._alarm("Yawning detected! Take a break!")

            self._calc_risk()

            # ── HUD
            color_map = {
                "SAFE"   : (0, 220, 120),
                "WARNING": (0, 200, 255),
                "DANGER" : (0, 50,  255)
            }
            col = color_map[self.alert_level]

            cv2.rectangle(frame, (0, 0), (w, 40), (15, 15, 25), -1)
            cv2.putText(frame, f"STATUS: {self.alert_level}", (10, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            cv2.putText(frame, f"RISK: {self.risk_score}%", (w-130, 27),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

            cv2.rectangle(frame, (0, h-45), (w, h), (15, 15, 25), -1)
            cv2.putText(frame, f"Blinks:{self.total_blinks}", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
            cv2.putText(frame, f"Yawns:{self.total_yawns}", (130, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
            cv2.putText(frame, f"FPS:{self.fps:.0f}", (250, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)

            if self.alert_level == "DANGER":
                overlay = frame.copy()
                cv2.rectangle(overlay, (0,0), (w,h), (0,0,180), -1)
                cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
                cv2.putText(frame, "! WAKE UP !", (w//2-130, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0,50,255), 3)

            if self.alert_level == "WARNING":
                cv2.putText(frame, "YAWNING DETECTED", (w//2-150, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,255), 2)

            # FPS
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                self.fps      = fps_counter / (time.time() - fps_start)
                fps_counter   = 0
                fps_start     = time.time()

            self._frame = frame

        cap.release()