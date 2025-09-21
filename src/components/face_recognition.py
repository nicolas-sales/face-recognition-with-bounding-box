from deepface import DeepFace
import pandas as pd
import cv2
import os
from pathlib import Path

class Recognizer:
    def __init__(
            self,
            db_path: str = "img_real",
            model_name: str = "Facenet",
            temp_dir: str = "tmp",
            threshold : float = 0.7   # plus petit = plus strict
    ):
            self.db_path = str(Path(db_path).resolve()) # resolve() -> Fige le chemin de la base : il devient indépendant de l’endroit d’où le script est éxécuté.
            self.model_name = model_name
            self.threshold = float(threshold)
            # Convertit temp_dir en Path et crée le dossier s’il n’existe pas (comme mkdir -p)
            self.temp_dir = Path(temp_dir).resolve()
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            # self.temp_path = str(self.temp_dir / "temp_face.jpg")
            self.temp_path = str(self.temp_dir / "temp_face.jpg")

    def identify(self, face_crop) -> str:
        # Prend un crop visage (BGR OpenCV) et renvoie :
        # - le nom (dossier parent ou nom de fichier) si trouvé sous le seuil,
        # - 'Inconnu' sinon,
        # - 'Non détecté' si DeepFace ne trouve aucun visage.

        try:
             # On écrit l’image du visage sur disque parce que DeepFace.find ne prend pas un ndarray directement, il attend un chemin.
             cv2.imwrite(self.temp_path, face_crop)

             dfs = DeepFace.find(
             img_path=self.temp_path,
             db_path=self.db_path,
             model_name=self.model_name,
             enforce_detection=False,  # True si detction de visage sinoin false
             silent=True) # pas de logs bruyants.

             # Retourne une liste de DataFrame (un par backend de détection). On utilise le premier.

             # sécurité : vérifie la liste renvoyée
             if not dfs or len(dfs) == 0:
                  return "Inconnu" # Si DeepFace n’a rien renvoyé, on n’a aucune correspondance -> “Inconnu”.
             
             # DeepFace.find(...) peut tester plusieurs backends de détection de visages
             # Exemples de backends : opencv, retinaface, mtcnn, ssd, dlib...
             # Chaque backend détecte/recadre le visage un peu différemment -> ça produit un DataFrame de résultats (identités + distances) par backend.
             # La fonction peut renvoyer une liste de DataFrames : [df_opencv, df_retinaface, df_mtcnn, ...]
             # dfs[0] est choisi par simplicité, car le premier correspond en général au backend par défaut

             df: pd.DataFrame = dfs[0] # On récupère le premier DataFrame
             if df is None or df.empty:
                  return "Inconnu" # S’il est vide : aucune image de la base n’est jugée proche -> “Inconnu”.

            # prend la meilleure correspondance et applique le seuil
             df = df.sort_values("distance", ascending=True) # Trie les résultats par distance croissante (meilleure correspondance en premier).
             best = df.iloc[0] # la plus petite distance
             if float(best["distance"]) > self.threshold:
                  return "Inconnu" # si la distance est > threshold, on considère que ce n’est pas la personne -> “Inconnu”.

            # Extrait le nom selon la convention de fichiers: nico_face.jpg -> "nico"
             identity_path = Path(best["identity"]) # chemin du fichier de la base qui a matché.
             base = identity_path.stem  # .stem donne le nom de fichier sans extension (ex. nico_face)
             name = base.split("_")[0]  # -> "nico"

             return name

        except Exception:
             return "Non détecté"