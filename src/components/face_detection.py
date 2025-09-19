from ultralytics import YOLO
import cv2
import os
import sys # Sert à récupérer des infos système (ici, le traceback) quand on construit des exceptions personnalisées
from src.exception import CustomException
from src.logger import logging

class FaceDetector:
    def __init__(self,model_path: str ="yolov8n-face.pt", device: str = "cuda", conf: float = 0.25): # chemin du modèle YOLO, l’appareil d’exécution (cuda ou cpu), et le seuil de confiance.
        try:

            import torch

            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # si on demande cuda mais qu'il n'y en a pas → fallback CPU
            if str(device).startswith("cuda") and not torch.cuda.is_available():
                device = "cpu"

            self.model = YOLO(model_path)
            self.device = device
            self.conf = conf
            logging.info("FaceDetector ready (model=%s, device=%s, conf=%.2f)", model_path, device, conf)
        except Exception as e:
            raise CustomException(e, sys)
        
    def detect(self, frame):

        # Détecte les visages sur un frame BGR (numpy array OpenCV)
        # Retourne une liste de tuples (x1, y1, x2, y2)

        try:
            results = self.model.predict(
                frame, device=self.device, conf=self.conf, verbose=False
            )
            boxes = results[0].boxes  # On récupère toutes les boxes (bounding boxes)

            boxes_xyxy = []

            for box in boxes:  # Boucle sur chaque personne détectée
                cls_id = int(box.cls[0])  # Pour chaque box détectée, on récupère la classe détectée (cls_id)
                label = self.model.names[cls_id] # Convertit l’ID en étiquette lisible via le mapping du modèle (ex. "face")
                #if label != "person":  # Si ce n’est pas une personne, on continue
                if label != "face":  # Si ce n’est pas un visage, on continue
                    continue
        
                # Extraction de la zone contenant la personne
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extraction des coordonnées du rectangle (xyxy)
                # box.xyxy[0] renvoie 4 nombres (souvent des floats) → on les convertit en int (pixels)
                boxes_xyxy.append((x1,y1,x2,y2))
            return boxes_xyxy
        
        except Exception as e:
            raise CustomException(e, sys)
