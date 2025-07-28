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

2. Chaque personne détectée est recadrée (person_crop)

3. person_crop est temporairement enregistrée comme image (temp_face.jpg)

4. DeepFace recherche une correspondance dans img_real/ avec le modèle Facenet

5. Si une correspondance est trouvée, le prénom est extrait et affiché

6. Sinon, "Inconnu" ou "Non détecté" s'affiche
"# face-recognition-with-bounding-box" 
