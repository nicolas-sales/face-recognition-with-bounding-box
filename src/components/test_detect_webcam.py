import cv2
import torch
from src.components.face_detection import FaceDetector

device = "cuda:0" if torch.cuda.is_available() else "cpu"

det = FaceDetector(model_path="yolov8n-face.pt", device=device, conf=0.25)

cap = cv2.VideoCapture(0) # 0 = caméra par défaut
if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam")

while True:
    ret, frame = cap.read()   # Lit une image (frame) depuis la webcam à chaque itération
    if not ret:  # ret est True si la capture a réussi sinon stop
        break

    boxes_xyxy = det.detect(frame)

    for (x1, y1, x2, y2) in boxes_xyxy:

        # Ajout d’une marge autour du visage
        margin = 10
        h, w, _ = frame.shape  # taille de l'image
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Webcam - Détection visages", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()