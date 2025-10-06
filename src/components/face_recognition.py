from deepface import DeepFace
import pandas as pd
import cv2
import os
from pathlib import Path
import tempfile

class Recognizer:
    def __init__(
        self,
        db_path: str = "img_real",
        model_name: str = "Facenet",
        temp_dir: str = "tmp",
        threshold: float = 0.7,  # plus petit = plus strict
        detector_backend: str = "retinaface",  # plus robuste
    ):
        self.db_path = str(Path(db_path).resolve())
        self.model_name = model_name
        self.threshold = float(threshold)
        self.detector_backend = detector_backend

        # dossier pour éventuels fichiers temp (même si on utilise NamedTemporaryFile)
        self.temp_dir = Path(temp_dir).resolve()
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def identify(self, face_crop) -> str:
        """
        Prend un crop visage (BGR OpenCV) et renvoie :
        - le nom (dossier parent ou nom de fichier) si match sous le seuil,
        - 'Inconnu' si pas de match,
        - 'Non détecté' si DeepFace ne détecte pas de visage dans le crop.
        """
        # fichier temporaire unique (évite les collisions en multi-req)
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir=str(self.temp_dir))
        os.close(fd)  # on ne garde pas le descripteur ouvert

        try:
            # Écrit le crop sur disque (DeepFace.find attend un chemin d'image)
            cv2.imwrite(tmp_path, face_crop)

            dfs = DeepFace.find(
                img_path=tmp_path,
                db_path=self.db_path,
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # on crope déjà avec YOLO
                silent=True,
            )

            # sécurité : liste vide
            if not dfs or len(dfs) == 0:
                return "Inconnu"

            df: pd.DataFrame = dfs[0]
            if df is None or df.empty:
                return "Inconnu"

            # trier par distance croissante
            if "distance" not in df.columns:
                return "Inconnu"
            df = df.sort_values("distance", ascending=True)
            best = df.iloc[0]
            if float(best["distance"]) > self.threshold:
                return "Inconnu"

            # Extraire le nom depuis le dossier parent si la base est organisée en sous-dossiers
            identity_path = Path(best["identity"])
            db_root_name = Path(self.db_path).resolve().name
            parent = identity_path.parent.name

            if parent.lower() != db_root_name.lower():
                # cas recommandé: img_real/<person>/fichier.jpg
                name = parent
            else:
                # fallback: base à plat -> prefix "nom_*.jpg"
                name = identity_path.stem.split("_")[0]

            return name

        except Exception:
            # ex: si DeepFace n'a vraiment détecté aucun visage
            return "Non détecté"
        finally:
            # nettoyage du fichier temporaire
            try:
                os.remove(tmp_path)
            except Exception:
                pass