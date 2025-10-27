from src.components.face_recognition import Recognizer
import cv2
from typing import List

class FaceRecognitionPipeline:
    def __init__(self, db_path: str = "img_real", model_name: str = "Facenet", threshold: float = 0.4, detector_backend: str = "retinaface"): # plus robuste
        self.rec = Recognizer(db_path=db_path, model_name=model_name, threshold=threshold, detector_backend=detector_backend)

    def identify(self, crops: List) -> List[str]:
        return [self.rec.identify(crop) for crop in crops]
    
    def identify_one(self, crop) -> str:
        return self.rec.identify(crop)
    
    def identify_one_with_score(self, crop):
        return self.rec.identify_with_score(crop)
    

if __name__=="__main__":

    IMG = r"C:\Users\nico_\Desktop\Reconnaissance faciale\image_test\image.jpg"

    img = cv2.imread(IMG)

    if img is None:
        raise FileNotFoundError("Image introuvable: {IMG}")

    pipe = FaceRecognitionPipeline(db_path= "img_real", model_name= "Facenet", threshold= 0.4)
    name = pipe.identify_one(img)
    print("Résultat:", name)