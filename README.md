# Système de Reconnaissance Faciale en Temps Réel avec YOLOv8 + DeepFace

Ce projet combine **YOLOv8** pour détecter les personnes dans une vidéo en temps réel (webcam),  
et **DeepFace** pour identifier les visages dans les zones détectées.


## Fonctionnalités

- Détection de personnes via webcam avec YOLOv8  
- Reconnaissance faciale avec DeepFace + modèle 'Facenet' (distance = 0.1299)
- Affichage du prénom si une correspondance est trouvée dans la base d’images  
- Affichage "Inconnu" ou "Non détecté" si aucune correspondance n’est trouvée  
- Accélération GPU (CUDA) pour des performances fluides  
- Gestion robuste des erreurs (ex : visage absent, photo de mauvaise qualité)  


## Pré-requis

- Python 3.10 recommandé  
- GPU NVIDIA avec CUDA  
- OS : Windows  


## Modules à installer


pip install ultralytics
pip install deepface
pip install opencv-python


## Si erreur avec torchvision sur CUDA :

pip install torchvision --index-url https://download.pytorch.org/whl/cu118


## Fonctionnement du script :

1. YOLOv8 détecte les personnes dans la webcam

2. Chaque personne détectée est recadrée (face_crop)

3. face_crop est temporairement enregistrée comme image (temp_face.jpg)

4. DeepFace recherche une correspondance dans img_real/ avec le modèle Facenet

5. Si une correspondance est trouvée, le prénom est extrait et affiché

6. Sinon, "Inconnu" ou "Non détecté" s'affiche


## Lancement du serveur

uvicorn app:app --reload --host 0.0.0.0 --port 8000

## URL host

http://127.0.0.1:8000/docs


# Pour le GPU

pip uninstall -y torch torchvision torchaudio
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

verification : 
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# Docker
requirements-docker.txt pour ne pas téléchargé dans l'image des fichiers trop lourd comme cuda

lancement:
docker build -t face-app:latest .
docker run -p 8000:8000 face-app:latest
http://localhost:8000/docs

# CI/CD GitHub Action
Une fois le workflows validé dans GitHub action:
docker login ghcr.io -u nicolas-sales
password: ...
Prendre l'image : docker pull ghcr.io/nicolas-sales/face-app:latest
Lancer l'application : docker run -p 8000:8000 ghcr.io/nicolas-sales/face-app:latest
http://localhost:8000/docs

