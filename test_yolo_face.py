from ultralytics import YOLO
from deepface import DeepFace
import cv2
import os

# Charger YOLO et DeepFace
model = YOLO("yolov8s.pt")
db_path = "img_real"
deepface_model = "Facenet"

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Détection avec YOLO
    results = model.predict(frame, device="cuda")
    boxes = results[0].boxes

    for box in boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]
        if label != "person":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_crop = frame[y1:y2, x1:x2]    # Extraction de la box
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, person_crop)

        try:
            df = DeepFace.find(
            img_path=temp_path,
            db_path=db_path,
            model_name=deepface_model,
            enforce_detection=True,  
            silent=True)

            df = df[0]
            if not df.empty:
                full_name = os.path.basename(df.iloc[0]["identity"]).split(".")[0]
                name = full_name.split("_")[0]
            else:
                name = "Inconnu"
        except Exception as e:
            name = "Non détecté"
            print(f"Aucun visage détecté : {e}")


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.imshow("YOLOv8 + DeepFace", frame)
    print("Image affichée")  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
