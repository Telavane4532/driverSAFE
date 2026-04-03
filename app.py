from flask import Flask, render_template, Response, jsonify
from flask_socketio import SocketIO
import cv2
import time
import threading
import base64
from detector_core import DrowsinessCore

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

core = DrowsinessCore()
threading.Thread(target=core.start, daemon=True).start()

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/stats")
def stats():
    return jsonify(core.get_stats())

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = core.get_frame()
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame)
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("[INFO] Dashboard running at http://localhost:5000")
    socketio.run(app, debug=False, port=5000)