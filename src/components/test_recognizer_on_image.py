import cv2
from src.components.face_recognition import Recognizer

# Configuration
rec = Recognizer(
    db_path="img_real",
    model_name="Facenet",
    threshold=0.7
)

# Chemin d'une image test
IMG = r"C:\Users\nico_\Desktop\Reconnaissance faciale\image_test\image.jpg"

img = cv2.imread(IMG)

if img is None:
    raise FileNotFoundError("Image introuvable: {IMG}")

print("Résultat:", rec.identify(img))


#name = rec.identify(img)

#print("Resultat:", name)