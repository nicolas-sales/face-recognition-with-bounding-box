FROM python:3.11-slim

WORKDIR /app

# dépendances système nécessaires à opencv et cie
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# on copie le requirements
COPY requirements-docker.txt .

# 1. installer pip à jour
RUN pip install --no-cache-dir --upgrade pip

# 2. installer d'abord torch CPU explicitement
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. ensuite installer le reste (deepface, ultralytics, etc.)
#    --no-deps = n'installe pas les dépendances automatiques supplémentaires.
#    car deepface/ultralytics vont essayer de réinstaller torch GPU, on veut bloquer ça.
RUN pip install --no-cache-dir --no-deps -r requirements-docker.txt

# 4. maintenant on installe les dépendances manquantes "générales" (celles qui ne sont pas torch)
#    Ici on laisse pip résoudre les libs normales genre opencv, numpy, pandas etc qu'on n'a pas encore.
#    Comme torch est déjà installé, pip ne va pas le remplacer.
RUN pip install --no-cache-dir -r requirements-docker.txt

# copie du code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
