# Dockerfile del backend NeuroAge — costruito dalla ROOT del progetto:
#   docker build -f backend/Dockerfile -t neuroage-backend .
#
# I modelli addestrati (outputs/*.pkl, *.pth) e i dati (data/) NON sono nel
# repo (vedi .gitignore): vanno montati come volume a runtime, altrimenti
# il server parte ma risponde 503 su /predict e /mesh con Grad-CAM.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY backend ./backend

RUN pip install --no-cache-dir .

ENV BRAIN_AGE_HOME=/app
EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
