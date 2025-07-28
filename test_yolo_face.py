from ultralytics import YOLO
from deepface import DeepFace
import cv2
import os

# Charger YOLO et DeepFace
#model = YOLO("yolov8s.pt")
model = YOLO("yolov8n-face.pt")
db_path = "img_real"
deepface_model = "Facenet"

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 = caméra par défaut

while True:
    ret, frame = cap.read()   # Lit une image (frame) depuis la webcam à chaque itération
    if not ret:  # ret est True si la capture a réussi sinon stop
        break

    # Détection avec YOLO
    results = model.predict(frame, device="cuda")
    boxes = results[0].boxes  # On récupère toutes les boxes (bounding boxes)

    for box in boxes:  # Boucle sur chaque personne détectée
        cls_id = int(box.cls[0])  # Pour chaque box détectée, on récupère la classe détectée (cls_id)
        label = model.names[cls_id]
        #if label != "person":  # Si ce n’est pas une personne, on continue
        if label != "face":  # Si ce n’est pas un visage, on continue
            continue
        
        # Extraction de la zone contenant la personne
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extraction des coordonnées du rectangle (xyxy)

        # Ajout d’une marge autour du visage
        margin = 10
        h, w, _ = frame.shape  # taille de l'image
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(w, x2 + margin)
        y2 = min(h, y2 + margin)

        #person_crop = frame[y1:y2, x1:x2]
        face_crop = frame[y1:y2, x1:x2]    # Découpage de frame pour ne garder que la zone person_crop
        temp_path = "temp_face.jpg"  # Sauvegarde de cette image dans temp_face.jpg (car DeepFace find() attend un chemin)
        #cv2.imwrite(temp_path, person_crop)
        cv2.imwrite(temp_path, face_crop)


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
