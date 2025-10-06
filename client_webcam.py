import cv2
import requests
from collections import deque, Counter

API_URL = "http://127.0.0.1:8000/infer"  # remplacement par "http://<IP-PUBLIC-EC2>:8000/infer" en prod et ouvrir le port 8000 dans le Security Group.
CAM_IDX = 0
MARGIN = 12
TIMEOUT = 10

def infer_frame(frame_bgr, margin=MARGIN):
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    if not ok:
        return []
    files = {"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}
    data = {"margin": str(margin)}
    r = requests.post(API_URL, files=files, data=data, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json().get("results", [])

# Classe anti-clignotement 
class NameSmoother:

    # Maintient un historique glissant des derniers noms reconnus
    # et stabilise l’affichage en évitant les alternances rapides.
    
    def __init__(self, window=5, require=3):
        self.buf = deque(maxlen=window)  # on garde les N derniers noms
        self.require = require           # nombre min d’occurrences pour valider

    def update(self, name: str) -> str:
        self.buf.append(name)
        n, c = Counter(self.buf).most_common(1)[0]
        # si le nom apparaît assez souvent récemment, on l’affiche
        return n if c >= self.require else "Inconnu"
    

def main():
    cap = cv2.VideoCapture(CAM_IDX)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")

    win = "Webcam -> FastAPI"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        try:
            results = infer_frame(frame)
        except Exception as e:
            cv2.putText(frame, f"API error: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            results = []

        for item in results:
            x1, y1, x2, y2 = item["box"]
            name = item["name"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, max(0, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
