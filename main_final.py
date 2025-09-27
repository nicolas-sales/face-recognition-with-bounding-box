import cv2
import torch

from src.pipeline.face_detection_pipeline import FaceDetectionPipeline
from src.pipeline.face_recognition_pipeline import FaceRecognitionPipeline

def main():
    # device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # pipelines
    det = FaceDetectionPipeline(model_path = "yolov8n-face.pt", device = device, conf = 0.25)
    rec = FaceRecognitionPipeline(db_path = "img_real", model_name = "Facenet", threshold = 0.8)

    # Webcam
    cam_idx=0
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")
    
    margin = 20
    
    while True:
        ret, frame = cap.read()   # Lit une image (frame) depuis la webcam à chaque itération
        if not ret:  # ret est True si la capture a réussi sinon stop
            break

        h, w, _ = frame.shape  # taille de l'image
        boxes_xyxy = det.detect_boxes(frame)

        for (x1, y1, x2, y2) in boxes_xyxy:

                # Ajout d’une marge autour du visage
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                # crop_visage -> pipeline de reconnaissance
                face_crop = frame[y1:y2, x1:x2]
                name = rec.identify_one(face_crop)

                # annotation
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                
        cv2.imshow("Détection + reconnaissance",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
     main()